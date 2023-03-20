import random
import re

import numpy as np

from mindspore.dataset import vision as vision
from mindspore.dataset.vision import Inter
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import RandomColorAdjust

__all__ = ['SVTRRecAug']

class SVTRGeometry(object):
    def __init__(self,
                 aug_type=0, 
                 degrees=15, 
                 translate=0.3,
                 scale=(0.5, 2.),
                 shear_x = 45,
                 shear_y = 15,
                 distortion=0.5,
                 p=0.5):
        
        self.p = p
        self.transforms = []
        self.aug_type = aug_type
        self.transforms.append(vision.RandomRotation(degrees=(degrees,degrees)))
        self.transforms.append(vision.RandomAffine(degrees=degrees, translate=(translate, translate), scale=scale, shear=(shear_x,shear_x,shear_y,shear_y)))
        self.transforms.append(vision.RandomPerspective(distortion_scale=distortion))


    def __call__(self, img):
        # no replacement when using weighted choice
        if random.random() < self.p:
            if self.aug_type:
                random.shuffle(self.transforms)
                transforms = Compose(self.transforms[:random.randint(1,3)])
                img = transforms(img)
            else:
                img = vision.ToPIL()(img)
                img = self.transforms[random.randint(0,2)](img)
            return img
        else:
            return img
        
def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)

class SVTRDeterioration(object):
    def __init__(self, var, factor, p=0.5):
        self.p = p
        transforms = []
        if var is not None:
            transforms.append(vision.GaussianBlur(3,var))
        if factor is not None:
            factor = round(sample_uniform(0, factor))
            transforms.append(vision.Rescale(rescale=factor,shift=-1.0))
        self.transforms = transforms

    def __call__(self, img):
        if random.random() < self.p:
            random.shuffle(self.transforms)
            transforms = Compose(self.transforms)
            return transforms(img)
        else:
            return img
        
class CVColorJitter(object):
    def __init__(self,
                 brightness=0.5,
                 contrast=0.5,
                 saturation=0.5,
                 hue=0.1,
                 p=0.5):
        self.p = p
        self.transforms = RandomColorAdjust(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)

    def __call__(self, img):
        if random.random() < self.p: return self.transforms(img)
        else: return img

class SVTRRecAug(object):
    def __init__(self,
                 aug_type=0,
                 geometry_p=0.5,
                 deterioration_p=0.25,
                 colorjitter_p=0.25,
                 **kwargs):
        self.transforms = Compose([
            SVTRGeometry(
                aug_type=aug_type,
                degrees=45,
                translate=0.0,
                scale=(0.5, 2.),
                shear_x=45,
                shear_y=15,
                distortion=0.5,
                p=geometry_p), SVTRDeterioration(
                    var=20,factor=4, p=deterioration_p),
            CVColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
                p=colorjitter_p)
        ])

    def __call__(self, data):
        img = data['image']
        img = self.transforms(img)
        data['image'] = img
        return data