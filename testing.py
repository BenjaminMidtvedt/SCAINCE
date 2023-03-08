# %%

from models.RPAE import RPAE

model = RPAE()
print(model)
#%%
import deeptrack as dt
import torch
import numpy as np
import matplotlib.pyplot as plt


image = dt.Value(lambda: np.zeros((3, 256, 256)))

def add_rectangle(x, y, w, h):
    def inner(image):
        _x = int(x)
        _y = int(y)
        _w = int(w)
        _h = int(h)
        image[:, _x:_x+_w, _y:_y+_h] = 1
        return image
    return inner

image >>= dt.Lambda(
    add_rectangle,
    x=lambda: np.random.randint(0, 256),
    y=lambda: np.random.randint(0, 256),
    w=lambda: np.random.randint(20, 50),
    h=lambda: np.random.randint(20, 50), 
) ^ 10

im = image()

plt.imshow(im[0])

# image = torch.zeros(1, 3, 256, 256)

boxes, scores = model(torch.Tensor(im[None]))

# %%
# For each box, crop the image 
import torch.nn.functional as F
from torchvision.ops import roi_align

crops = []
# for box in boxes[0][:1]:
    # size = (int(box[3] - box[1]), int(box[2] - box[0]))
    # print(box)
crops = roi_align(
    torch.Tensor(im[None]),
    boxes,
    (32, 32),
)

print(crops.shape)

        
# model.training_step(image, 0)

# %%

# apply nms to the boxes
from torchvision.ops import nms

keep = nms(boxes[0], scores[0], 0.5)

kept_boxes = boxes[keep]
kept_scores = scores[keep]

kept_boxes = [kept_boxes]
kept_scores = [kept_scores]

plt.imshow(im[0])
for box in kept_boxes[0]:
    box = box 

    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=1))



# %%
torch.max(boxes[0])
# %%

xx = torch.Tensor(im[None])
xx[0] = 1