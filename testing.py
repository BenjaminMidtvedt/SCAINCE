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

model = RPAE()
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
    ^ 20
)

timages = [image.update()()._value.astype(float) for _ in range(64)]

# image = torch.zeros(1, 3, 256, 256)

# boxes, scores = model(torch.Tensor(im[None]))

# %%


input_tensor = torch.Tensor(np.array(timages))
# resize
input_tensor = F.interpolate(input_tensor, size=(64, 64))


# %%

trainer = pl.Trainer(gpus=1, max_epochs=10, log_every_n_steps=1)
trainer.fit(
    model.background_autoencoder,
    torch.utils.data.DataLoader(input_tensor.float(), batch_size=4),
)

# %%

output = model.background_autoencoder(input_tensor)
print(output.shape)

for x, y in zip(input_tensor[:4], output):

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(x.permute(1, 2, 0).detach().numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(y.permute(1, 2, 0).detach().numpy())

    plt.subplot(1, 3, 3)
    plt.imshow(
        np.abs((x - y).permute(1, 2, 0).detach().numpy()[..., 0]),
        cmap="jet",
        vmin=-1,
        vmax=1,
    )
    plt.colorbar()
    plt.show()

# %%


trainer = pl.Trainer(gpus=1, max_epochs=100, log_every_n_steps=1)
trainer.fit(model, torch.utils.data.DataLoader(timages, batch_size=4))


# %%
# For each box, crop the image


trainer = pl.Trainer(gpus=1, max_epochs=0, log_every_n_steps=1)
trainer.fit(model, torch.utils.data.DataLoader(timages, batch_size=4))

imagess = torch.Tensor(timages[:4])
boxess, scoress = model(imagess)

for i in range(4):
    images = imagess[i : i + 1]
    boxes = boxess[i : i + 1]
    scores = scoress[i : i + 1]

    # print(boxes[0].shape)
    pooled_boxes = ops.roi_align(images, boxes, output_size=(28, 28), spatial_scale=1.0)

    # n_reconstruct =
    print(boxes[0].shape)

    reconstructed_images = torch.zeros_like(images)
    image_weights = torch.zeros_like(images) + 1e-8
    for i in range(pooled_boxes.shape[0]):

        box = boxes[0][i]
        pooled_box = pooled_boxes[i]

        # Resize and add the pooled box to the image using grid_sample
        scale_x = 1024 / (box[2] - box[0])
        scale_y = 1024 / (box[3] - box[1])

        half_box_width = (box[2] - box[0]) / 2
        half_box_height = (box[3] - box[1]) / 2
        translate_x = -((box[0] + half_box_width) / 512 - 1) * scale_x
        translate_y = -((box[1] + half_box_height) / 512 - 1) * scale_y
        scale_and_translate = torch.Tensor(
            [
                [scale_x, 0, translate_x],
                [0, scale_y, translate_y],
            ]
        )

        grid = F.affine_grid(scale_and_translate[None], images.shape)
        resized_box = F.grid_sample(pooled_box[None], grid)[0]

        where_box = resized_box != 0
        # Add the resized box to the image
        reconstructed_images[0] += resized_box * scores[0][i]
        image_weights[0] += where_box * scores[0][i]

    reconstructed_images /= image_weights

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(images[0].detach().numpy().transpose(1, 2, 0))
    for box in boxes[0]:
        box = box.detach().numpy()

        plt.gca().add_patch(
            plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                edgecolor="red",
                linewidth=1,
            )
        )
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_images[0].detach().numpy().transpose(1, 2, 0))
    for box in boxes[0]:
        box = box.detach().numpy()

        plt.gca().add_patch(
            plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                edgecolor="red",
                linewidth=1,
            )
        )
    plt.show()
# %%

images = torch.Tensor(timages[:4])
boxes, scores = model(images)
small_images = F.interpolate(
    images, size=(64, 64), mode="bilinear", align_corners=False
)
autoencoder_output = model.background_autoencoder(small_images)

feature_autoencoder_input = ops.roi_align(
    images,
    boxes,
    output_size=(32, 32),
    spatial_scale=1.0,
)
boxes_per_image = [len(box) for box in boxes]
feature_autoencoder_output = model.feature_autoencoder(feature_autoencoder_input)

# get list of features for each image
feature_autoencoder_output_list = []
start = 0
for boxes_per_image in boxes_per_image:
    end = start + boxes_per_image
    feature_autoencoder_output_list.append(
        torch.abs(
            feature_autoencoder_output[start:end] - feature_autoencoder_input[start:end]
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
    )

    error_density_in_box = error_density_in_box.mean((1, 2, 3))

    error_mass_in_box = error_density_in_box * ops.boxes.box_area(boxes[batch_idx])

    feature_encoder_error = feature_autoencoder_output_list[batch_idx]
    feature_encoder_error = feature_encoder_error.mean((1, 2, 3))

    print(feature_encoder_error)

    # high values mean that the autoencoder improved the image
    # low values mean that the autoencoder did not improve the image
    improved_error_in_box = (error_density_in_box - feature_encoder_error).detach()

    # weighted sum of improved error and score
    # This should force the model to put a high score on boxes that it thinks will improve the image
    # and a low score on boxes that it thinks will not improve the image
    _weighted_score = (scores[batch_idx] * improved_error_in_box).view(-1).sum() / (
        scores[batch_idx].view(-1).sum() + 1e-8
    )

    # potential improvement to the image
    _weighted_potential_improvement = (error_mass_in_box * scores[batch_idx]).view(
        -1
    ).sum() / (scores[batch_idx].view(-1).sum() + 1e-8)
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
#%%

print(weighted_error, weighted_score, weighted_potential_improvement)
# %%
images = torch.Tensor(timages[:4])
images, _ = model.transform(images)

# get image features
features = model.backbone(images.tensors)
features = list(features.values())

# get proposals
objectness, pred_bbox_deltas = model.rpn.head(features)

objectness


# %%

# pseudo training code

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
            epoch_loss(),
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
            box,
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


pretrain_background_autoencoder(background_autoencoder, train_loader, 100)


# for epoch in range(epochs):

#     for batch_idx, images in enumerate(train_loader):

#         background = estimate_background(model, images)
#         background_error = get_background_error(images, background)

#         train_background_autoencoder(model, images, background_error)

# scores, boxes = get_proposals(model, images)
# box_error_density, box_error_mass = eval_boxes(boxes, background_error)
# box_score = eval_box_scores(box_error_density, box_error_mass)

# train_feature_autoencoder(model, images, boxes, box_score)

# %%
