# %%

from models.RPAE import RPAE

import deeptrack as dt
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from models.autoencoders import AutoEncoder
import torch.nn.functional as F
from torchvision.ops import roi_align

from torchvision.ops import nms
from torchvision import ops

image = dt.Value(lambda: np.zeros((3, 1024, 1024)))


def add_rectangle(x, y, w, h):
    def inner(image):
        _x = int(x)
        _y = int(y)
        _w = int(w)
        _h = int(h)
        image[:, _x : _x + _w, _y : _y + _h] = 1
        return image

    return inner


image >>= (
    dt.Lambda(
        add_rectangle,
        x=lambda: np.random.randint(100, 900),
        y=lambda: np.random.randint(100, 900),
        w=lambda: np.random.randint(25, 75),
        h=lambda: np.random.randint(25, 75),
    )
    ^ 10
)

timages = [image.update()()._value.astype(float) for _ in range(64)]


# %%

background_autoencoder = AutoEncoder((3, 64, 64), 128)
feature_autoencoder = AutoEncoder((3, 32, 32), 128)

background_optimizer = torch.optim.Adam(background_autoencoder.parameters(), lr=0.001)
feature_optimizer = torch.optim.Adam(feature_autoencoder.parameters(), lr=0.001)

train_loader = torch.utils.data.DataLoader(timages, batch_size=4, shuffle=True)


def train_background_autoencoder(model, images, background_error):
    images = images.float()
    model.zero_grad()

    small_images = F.interpolate(
        images, size=(64, 64), mode="bilinear", align_corners=False
    )
    pred_background = model(small_images)
    loss = F.l1_loss(pred_background, small_images)

    loss.backward()
    background_optimizer.step()

    return loss.item()


def pretrain_background_autoencoder(model, train_loader, epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, images in enumerate(train_loader):

            loss = train_background_autoencoder(model, images, None)
            epoch_loss += loss
        print(
            f"Pretraining background autoencoder {epoch + 1} / {epochs}:",
            epoch_loss,
        )


def estimate_background(model, images):
    small_images = F.interpolate(
        images, size=(64, 64), mode="bilinear", align_corners=False
    )
    pred_background = model(small_images)
    big_pred_background = F.interpolate(
        pred_background, size=(images.shape[2], images.shape[3]), mode="bilinear"
    )
    return big_pred_background


def get_background_error(images, background):
    return torch.abs(images - background)


def get_proposals(model, images):
    features = model.backbone(images.tensors)
    features = list(features.values())

    # get proposals
    objectness, pred_bbox_deltas = model.rpn.head(features)

    boxes = model.rpn.head.box_selector_test(
        [features[0].shape[-2:]],
        objectness,
        pred_bbox_deltas,
        images.image_sizes,
    )
    return objectness, boxes


def eval_boxes(boxes, background_error):

    box_error_density = []
    box_error_mass = []
    for batch_idx, box in enumerate(boxes):
        error_density_in_box = ops.roi_align(
            background_error[batch_idx : batch_idx + 1],
            [box],
            output_size=(1, 1),
            spatial_scale=1.0,
        )

        error_density_in_box = error_density_in_box.mean((1, 2, 3))

        error_mass_in_box = error_density_in_box * ops.boxes.box_area(box)

        box_error_density.append(error_density_in_box)
        box_error_mass.append(error_mass_in_box)

    return box_error_density, box_error_mass


def eval_box_scores(box_error_density, box_error_mass):

    box_score = []
    for batch_idx, (error_density, error_mass) in enumerate(
        zip(box_error_density, box_error_mass)
    ):
        score = error_mass * error_density
        box_score.append(score)

    return box_score


# %%
pretrain_background_autoencoder(background_autoencoder, train_loader, 50)

# %%
input_images = torch.Tensor(timages[:4]).float()
predicted_backgrounds = estimate_background(background_autoencoder, input_images)
background_error = get_background_error(input_images, predicted_backgrounds)

# %%

rpn = RPAE(
    rpn_pre_nms_top_n_train=1000,
    rpn_pre_nms_top_n_test=1000,
    rpn_post_nms_top_n_train=1000,
    rpn_post_nms_top_n_test=1000,
)

boxes, scores = rpn(input_images)

# %%


def eval_box_scores(box_error_density, box_error_mass):

    a = 0.5
    b = 1 - a
    box_score = []
    for batch_idx, (error_density, error_mass) in enumerate(
        zip(box_error_density, box_error_mass)
    ):
        score = error_density**a * error_mass**b
        box_score.append(score)

    return box_score


density, mass = eval_boxes(boxes, background_error)
box_scores = eval_box_scores(density, mass)

all_boxes = []
all_scores = []

for batch_idx in range(4):

    batch_boxes = boxes[batch_idx]
    batch_scores = box_scores[batch_idx]
    batch_boxes_and_scores = zip(batch_boxes, batch_scores)
    batch_boxes_and_scores = sorted(
        batch_boxes_and_scores, key=lambda x: x[1], reverse=True
    )
    batch_boxes, batch_scores = zip(*batch_boxes_and_scores)

    # stack boxes and scores
    batch_boxes = torch.stack(batch_boxes)
    batch_scores = torch.stack(batch_scores)

    batch_boxes = batch_boxes[:250]
    batch_scores = batch_scores[:250]

    # Do nms
    keep = ops.boxes.nms(batch_boxes, batch_scores, 0.4)

    batch_boxes = batch_boxes[keep]
    batch_scores = batch_scores[keep]

    all_boxes.append(batch_boxes)
    all_scores.append(batch_scores)

    plt.figure(figsize=(10, 10))
    plt.imshow(input_images[batch_idx].permute(1, 2, 0))

    for box, score in zip(batch_boxes, batch_scores):
        # if box[0] > 300 or box[1] > 300:
        #     continue

        # # remove boxes larger than 200x200
        # if box[2] - box[0] > 200 or box[3] - box[1] > 200:
        #     continue

        box = box.detach().numpy()
        plt.plot(
            [box[0], box[2], box[2], box[0], box[0]],
            [box[1], box[1], box[3], box[3], box[1]],
            color="red",
        )
        plt.text(box[0], box[1], f"{score:.3f}", color="red")
    # plt.colorbar()
    plt.show()

# %%
from models.autoencoders import CNN_Encoder, CNN_Decoder


class STAutoEncoder(pl.LightningModule):
    def __init__(self, input_shape, latent_dim=64):
        super(AutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = CNN_Encoder(latent_dim, input_shape)

        _tensor = torch.rand(1, *input_shape)
        _conv_out = self.encoder.conv(_tensor)
        self.decoder = CNN_Decoder(
            latent_dim, self.encoder.flat_fts, _conv_out.shape[2], _conv_out.shape[3]
        )

        self.localization = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=7),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 10, kernel_size=5),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU(True),
        )

        _localization_out_shape = self.localization(_tensor).shape

        self.fc_loc = torch.nn.Sequential(
            torch.nn.Linear(
                10 * _localization_out_shape[2] * _localization_out_shape[3], 32
            ),
            torch.nn.ReLU(True),
            torch.nn.Linear(32, 4),
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0, 0, 0, 0], dtype=torch.float))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        return theta

    def forward(self, x):
        return self.decode(self.encode(x))

    def training_step(self, batch, batch_idx):
        x_hat = self(batch)
        loss = F.l1_loss(x_hat, batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


feature_autoencoder = STAutoEncoder((3, 32, 32))


residual = input_images - predicted_backgrounds

larger_boxes = []
for batch_idx, batch_boxes in enumerate(all_boxes):
    batch_boxes = batch_boxes.detach().clone()
    batch_boxes[:, 2] = batch_boxes[:, 2] * 2
    batch_boxes[:, 3] = batch_boxes[:, 3] * 2
    batch_boxes[:, 0] = batch_boxes[:, 0] - batch_boxes[:, 2] / 4
    batch_boxes[:, 1] = batch_boxes[:, 1] - batch_boxes[:, 3] / 4

    larger_boxes.append(batch_boxes)

padded_pooled_features = ops.roi_align(
    residual, larger_boxes, output_size=(32, 32), spatial_scale=1.0
)

feature_autoencoder_batch_size = 8

feature_autoencoder_optimizer = torch.optim.Adam(
    feature_autoencoder.parameters(), lr=5e-4
)

pooled_features = padded_pooled_features.detach()

for epoch in range(5):

    for f_batch in range(0, len(pooled_features), feature_autoencoder_batch_size):
        feature_autoencoder.train()
        feature_autoencoder_optimizer.zero_grad()

        input_features = pooled_features[
            f_batch : f_batch + feature_autoencoder_batch_size
        ]

        pred_bbox_delta = feature_autoencoder.stn(input_features)
        pred_bbox_delta = pred_bbox_delta * [[16, 16, 1, 1]]

        feature_autoencoder_loss = torch.abs(input_features - predicted_features).mean()

        feature_autoencoder_loss.backward()
        feature_autoencoder_optimizer.step()

        print(f"feature autoencoder loss: {feature_autoencoder_loss.item():.3f}")

# %%
residual = input_images - predicted_backgrounds
pooled_features = ops.roi_align(
    residual, all_boxes, output_size=(32, 32), spatial_scale=1.0
).detach()

predicted_features = feature_autoencoder(pooled_features)

concatenated_boxes = torch.cat(all_boxes)
concatenated_boxes_batch_idx = [[i] * len(all_boxes[i]) for i in range(len(all_boxes))]
concatenated_boxes_batch_idx = sum(concatenated_boxes_batch_idx, [])
feature_error_density = (
    torch.abs(pooled_features - predicted_features).abs().mean((1, 2, 3))
)

feature_error_mass = feature_error_density * ops.box_area(concatenated_boxes)

pooled_error_density = ops.roi_align(
    background_error, all_boxes, output_size=(1, 1), spatial_scale=1.0
).mean((1, 2, 3))

pooled_error_mass = pooled_error_density * ops.box_area(concatenated_boxes)

feature_box_score = eval_box_scores(feature_error_density, feature_error_mass)
pooled_box_score = eval_box_scores(pooled_error_density, pooled_error_mass)

improvement = [x - y for x, y in zip(pooled_box_score, feature_box_score)]
improvement = torch.stack(improvement)

# Find indexes of 10 largest improvement
sorted_improvement, improvement_idx = torch.sort(improvement.detach(), descending=True)


# %%
residual = input_images - predicted_backgrounds
residual_copy = residual.clone()
correction_map = torch.zeros_like(residual_copy)
score_map = torch.zeros_like(residual_copy[:, 0:1, :, :]) + 1e-3

error = torch.abs(residual_copy).mean((1, 2, 3))
selected_boxes = [[] for _ in range(len(input_images))]
for idx in improvement_idx:

    batch_idx = concatenated_boxes_batch_idx[idx]
    box = concatenated_boxes[idx].detach()
    score = improvement[idx]
    image = residual_copy[batch_idx]
    preducted_feature = predicted_features[idx]

    if score < 0:
        break

    # Remove box from image
    # Resize and add the pooled box to the image using grid_sample
    scale_x = image.shape[1] / (box[2] - box[0])
    scale_y = image.shape[2] / (box[3] - box[1])

    image_half_width = image.shape[1] / 2
    image_half_height = image.shape[2] / 2
    half_box_width = (box[2] - box[0]) / 2
    half_box_height = (box[3] - box[1]) / 2
    translate_x = -((box[0] + half_box_width) / image_half_width - 1) * scale_x
    translate_y = -((box[1] + half_box_height) / image_half_height - 1) * scale_y
    scale_and_translate = torch.Tensor(
        [
            [scale_x, 0, translate_x],
            [0, scale_y, translate_y],
        ]
    )

    grid = F.affine_grid(scale_and_translate[None], image[None].shape)
    resized_box = F.grid_sample(preducted_feature[None], grid)[0]

    where_box = (resized_box != 0).any(0, keepdim=True)

    correction_map[batch_idx] += resized_box * score
    score_map[batch_idx] += where_box * score

    new_residual = (
        residual_copy[batch_idx] - correction_map[batch_idx] / score_map[batch_idx]
    )
    remaining_error_batch = torch.abs(new_residual).mean()
    print(remaining_error_batch)

    _improvement = error[batch_idx] - remaining_error_batch
    if _improvement > 0.001:
        error[batch_idx] = remaining_error_batch
        # residual_copy[batch_idx] = new_residual
        selected_boxes[batch_idx].append(box)

        print(f"score: {score:.3f}, error: {error[batch_idx]:.3f}")
    else:
        correction_map[batch_idx] -= resized_box * score
        score_map[batch_idx] -= where_box * score
    # break

    # where_box = resized_box != 0
    # # Add the resized box to the image
    # reconstructed_images[0] += resized_box * scores[0][i]
    # image_weights[0] += where_box * scores[0][i]
# %%

pred = correction_map / score_map

for i in range(len(input_images)):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(residual[i, 0].detach().numpy())
    for box in selected_boxes[i]:
        plt.gca().add_patch(
            plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )
    plt.subplot(1, 2, 2)
    plt.imshow(pred[i, 0].detach().numpy())
    # plt.colorbar()
    plt.show()

# %%
