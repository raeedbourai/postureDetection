import os
import numpy as np 
import imageio.v3 as imageio
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

images = np.array(
    [imageio.imread('images/frame66.jpg') for _ in range(32)],
    dtype=np.uint8
)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    # iaa.Crop(percent=(0, 0.3)),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=15.0)),
    iaa.LinearContrast((0.75, 1.5)),
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # iaa.Affine(
    #     # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    # )
], random_order=True)

images_aug = seq(images=images)

print("Augmented:")
plt.imshow(ia.draw_grid(images_aug[:8], cols=4, rows=2))
plt.axis('off')
plt.show()