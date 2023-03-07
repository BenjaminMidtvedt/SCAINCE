import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torchvision.models as models
from torchvision.models.detection import rpn, anchor_utils, transform
from torchvision import ops


class RPAE(pl.LightningModule):
    """Region Proposal Autoencoder"""

    def __init__(
        self,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_score_thresh=0.0,
    ):

        self.backbone = models.resnet18()

        out_channels = self.backbone.out_channels

        anchor_generator = anchor_utils.AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
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

        self.transform = transform.GeneralizedRCNNTransform(
            min_size=800, max_size=1333, image_mean=[0.485], image_std=[0.229]
        )

    def forward(self, images):

        # if number of channels is 1, convert to 3 channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        images = self.transform(images)

        # get image features
        features = self.backbone(images)

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

        proposals = self.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.rpn.filter_proposals(
            proposals, objectness, images.image_sizes, num_anchors_per_level
        )

        return boxes, scores

    def training_step(self, images, _batch_idx):
        boxes, scores = self(images)

        pooled_boxes = ops.roi_align(
            images, boxes, output_size=(1, 1), spatial_scale=1.0 / 16
        )

        boxes_area = (boxes[:, :, 2] - boxes[:, :, 0]) * (
            boxes[:, :, 3] - boxes[:, :, 1]
        )

        boxes_weight = boxes_area * pooled_boxes

        return {"loss": 0}
