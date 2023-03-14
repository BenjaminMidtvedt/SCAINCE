import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torchvision.models as models
from torchvision.models.detection import rpn, anchor_utils, transform, backbone_utils
from torchvision import ops

from .autoencoders import AutoEncoder


class RPAE(pl.LightningModule):
    """Region Proposal Autoencoder"""

    def __init__(
        self,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_pre_nms_top_n_train=100,
        rpn_pre_nms_top_n_test=100,
        rpn_post_nms_top_n_train=100,
        rpn_post_nms_top_n_test=100,
        rpn_nms_thresh=0.7,
        rpn_score_thresh=0.0,
    ):
        super().__init__()
        self.backbone = backbone_utils.resnet_fpn_backbone(
            "resnet18", None, trainable_layers=5
        )

        out_channels = self.backbone.out_channels

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = anchor_utils.AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test
        )

        rpn_head = rpn.RPNHead(
            out_channels, anchor_generator.num_anchors_per_location()[0]
        )

        self.rpn = rpn.RegionProposalNetwork(
            anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        self.background_autoencoder = AutoEncoder(
            input_shape=(3, 64, 64),
            latent_dim=256,
        )

        self.feature_autoencoder = AutoEncoder(
            input_shape=(3, 32, 32),
            latent_dim=256,
        )

        self.transform = transform.GeneralizedRCNNTransform(
            min_size=800, max_size=1333, image_mean=[0.485], image_std=[0.229]
        )

    def forward(self, images):

        # if number of channels is 1, convert to 3 channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, _ = self.transform(images)

        # get image features
        features = self.backbone(images.tensors)
        features = list(features.values())

        # get proposals
        objectness, pred_bbox_deltas = self.rpn.head(features)
        anchors = self.rpn.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [
            s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
        ]

        objectness, pred_bbox_deltas = rpn.concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )

        proposals = self.rpn.box_coder.decode(pred_bbox_deltas, anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(
            proposals, objectness, images.image_sizes, num_anchors_per_level
        )

        scaled_boxes = []
        for i, boxtensor in enumerate(boxes):
            original_image_size = original_image_sizes[i]
            image_size = images.image_sizes[i]
            scale_x = original_image_size[1] / image_size[1]
            scale_y = original_image_size[0] / image_size[0]

            scaled_boxtensor = boxtensor * torch.tensor(
                [scale_x, scale_y, scale_x, scale_y], device=boxtensor.device
            )

            scaled_boxes.append(scaled_boxtensor)

        return scaled_boxes, scores

    def training_step(self, images, _batch_idx):
        images = images.float()
        boxes, scores = self(images)

        # resize image to 64x64
        small_images = F.interpolate(
            images, size=(64, 64), mode="bilinear", align_corners=False
        )
        autoencoder_output = self.background_autoencoder(small_images)

        feature_autoencoder_input = ops.roi_align(
            images,
            boxes,
            output_size=(32, 32),
            spatial_scale=1.0,
        )
        boxes_per_image = [len(box) for box in boxes]
        feature_autoencoder_output = self.feature_autoencoder(feature_autoencoder_input)

        # get list of features for each image
        feature_autoencoder_output_list = []
        start = 0
        for boxes_per_image in boxes_per_image:
            end = start + boxes_per_image
            feature_autoencoder_output_list.append(
                torch.abs(
                    feature_autoencoder_output[start:end]
                    - feature_autoencoder_input[start:end]
                )
            )
            start = end

        # restored background image
        background = F.interpolate(
            autoencoder_output,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).detach()

        bg_error = torch.abs(images - background)

        loss = 0
        weighted_error = 0
        weighted_score = 0
        weighted_potential_improvement = 0

        for batch_idx in range(images.shape[0]):

            error = bg_error[batch_idx : batch_idx + 1]
            error_density_in_box = ops.roi_align(
                error,
                boxes[batch_idx : batch_idx + 1],
                output_size=(1, 1),
                spatial_scale=1.0,
            ).mean((1, 2, 3))

            error_mass_in_box = error_density_in_box * ops.boxes.box_area(
                boxes[batch_idx]
            )

            feature_encoder_error = feature_autoencoder_output_list[batch_idx]
            feature_encoder_error = feature_encoder_error.mean((1, 2, 3))

            # high values mean that the autoencoder improved the image
            # low values mean that the autoencoder did not improve the image
            improved_error_in_box = (
                error_density_in_box - feature_encoder_error
            ).detach()

            # weighted sum of improved error and score
            # This should force the model to put a high score on boxes that it thinks will improve the image
            # and a low score on boxes that it thinks will not improve the image
            _weighted_score = (scores[batch_idx] * improved_error_in_box).view(
                -1
            ).sum() / (scores[batch_idx].view(-1).sum() + 1e-8)

            # potential improvement to the image
            _weighted_potential_improvement = (
                error_mass_in_box * scores[batch_idx]
            ).view(-1).sum() / (scores[batch_idx].view(-1).sum() + 1e-8)
            # print(error_mass_in_box, scores[batch_idx])

            # weighted sum of feature encoder error and error mass
            # This should force the autoencoder to focus its learning on the areas of the image
            # with high error mass
            # The error mass is detached from the graph to avoid backpropagating through the
            # region proposal network
            detached_error_mass_in_box = error_density_in_box.detach()
            _weighted_error = (detached_error_mass_in_box * feature_encoder_error).view(
                -1
            ).sum() / (detached_error_mass_in_box.view(-1).sum() + 1e-8)

            # combine the two losses.
            # Weighted score should be maximized
            # Weighted error should be minimized
            weighted_error += _weighted_error
            weighted_score += _weighted_score
            weighted_potential_improvement += _weighted_potential_improvement

        weighted_error = weighted_error / images.shape[0]
        weighted_score = weighted_score / images.shape[0]
        weighted_potential_improvement = (
            weighted_potential_improvement / images.shape[0]
        )

        loss = weighted_error - weighted_score

        losses = {
            "loss": loss,
        }

        metrics = {
            "weighted_error": weighted_error,
            "weighted_score": weighted_score,
            "weighted_potential_improvement": weighted_potential_improvement,
        }

        self.log_dict(metrics, on_step=True, prog_bar=True)
        return losses

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def filter_proposals(
        self,
        proposals,
        objectness,
        image_shapes,
        num_anchors_per_level,
    ):

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self.rpn._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(
            proposals, objectness_prob, levels, image_shapes
        ):
            boxes = ops.boxes.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = ops.boxes.remove_small_boxes(boxes, self.rpn.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.rpn.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = ops.boxes.batched_nms(boxes, scores, lvl, self.rpn.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.rpn.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores
