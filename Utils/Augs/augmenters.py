from imgaug import augmenters as iaa
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class Augumentations:
        seg_raf = iaa.Sometimes(
                0.5,
                iaa.Sequential([iaa.Fliplr(p=0.5), iaa.Affine(rotate=(-25, 25))]),
                iaa.Sequential([iaa.RemoveSaturation(1),iaa.Affine(scale=(1.0, 1.05))]))
        seg_raftest2 = iaa.Sequential([iaa.RemoveSaturation(1),iaa.Affine(scale=(1.0, 1.05))])
        seg_raftest1 = iaa.Sequential([iaa.Fliplr(p=0.5), iaa.Affine(rotate=(-25, 25))])