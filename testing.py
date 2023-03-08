# %%

from models.RPAE import RPAE


#%%
import deeptrack as dt
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
model = RPAE()
image = dt.Value(lambda: np.zeros((3, 256, 256)))


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
        x=lambda: np.random.randint(0, 256),
        y=lambda: np.random.randint(0, 256),
        w=lambda: np.random.randint(20, 50),
        h=lambda: np.random.randint(20, 50),
    )
    ^ 10
)

images = [image.update()()._value.astype(float) for _ in range(256)]

# image = torch.zeros(1, 3, 256, 256)

# boxes, scores = model(torch.Tensor(im[None]))

# %%
import pytorch_lightning as pl


trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.fit(model, torch.utils.data.DataLoader(images, batch_size=4))


# %%
# For each box, crop the image
import torch.nn.functional as F
from torchvision.ops import roi_align

from torchvision.ops import nms
from torchvision import ops

im = image.update()()
images = torch.Tensor(im[None])
boxes, scores = model(images)
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
    scale_x = 256 / (box[2] - box[0])
    scale_y = 256 / (box[3] - box[1])

    half_box_width = (box[2] - box[0]) / 2
    half_box_height = (box[3] - box[1]) / 2
    translate_x = -((box[0] + half_box_width) / 128 - 1) * scale_x
    translate_y = -((box[1] + half_box_height) / 128 - 1) * scale_y
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
    reconstructed_images[0] += resized_box
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
# %%

boxes
