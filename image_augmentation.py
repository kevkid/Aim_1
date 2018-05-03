#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:34:09 2018

@author: kevin
image augmentation
The idea is to pass in an image and this will create an augmentation of that image
"""

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import imageio
import pandas as pd
from random import randint
#pass in single image, the width, then height
def augment_images(image_path, width = 64, height = 64, include_original_image = True):
    #ia.seed(1)
    image = imageio.imread(image_path)
    if len(image.shape) < 3:#if there is no channel data
        image = np.stack((image,)*3, -1)
    # Example batch of images.
    # The array has shape (32, 64, 64, 3) and dtype uint8.
    images = np.array(
        [image for _ in range(8)],
        dtype=np.uint8
    )
    
    seq = iaa.Sequential([
        iaa.Scale({"height": height, "width": width}),#scale the image
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order
    
    images_aug = seq.augment_images(images)
    if include_original_image == True:
        images_aug = np.vstack([images_aug, iaa.Sequential([iaa.Scale({"height": height, "width": width}),]).augment_images([image])])
    return images_aug

def no_augmentation(image_path, width = 64, height = 64, include_original_image = True):
    image = imageio.imread(image_path)
    if len(image.shape) < 3:#if there is no channel data
        image = np.stack((image,)*3, -1)
    return iaa.Sequential([iaa.Scale({"height": height, "width": width}),]).augment_images([image])

'''Sample size in percent'''
def get_average_shape(image_list, sample_size = 10): 
    random_images = [randint(0, len(image_list)) for i in range(len(image_list)//sample_size)]
    width = 0
    height = 0
    for i in random_images:
        img_shape = imageio.imread(image_list.iloc[i]["location"]).shape
        width += img_shape[1]
        height +=img_shape[0]
    width //= len(random_images)
    height //= len(random_images)
    return (width, height)

def augment_images_bulk(images, shape):
    array = pd.DataFrame(columns=['x', 'y'], dtype=np.int8)
    count = 0
    for index, image in images.iterrows():
        #print(image)
        aug_images = augment_images(image["location"], shape[0], shape[1])#augmented image array
        for aug_image in aug_images:#go through every augmented image and flatten it then append to list
            array.at[count] = {'x':aug_image.flatten(), 'y':image["class_id"]}
            count+=1
    return array

def no_augment_images_bulk(images, shape):
    array = pd.DataFrame(columns=['x', 'y'], dtype=np.int8)
    count = 0
    for index, image in images.iterrows():
        #print(image)
        aug_images = no_augmentation(image["location"], shape[0], shape[1])#augmented image array
        for aug_image in aug_images:#go through every augmented image and flatten it then append to list
            array.at[count] = {'x':aug_image.flatten(), 'y':image["class_id"]}
            count+=1
    return array

'''
    import matplotlib.pyplot as plt
    def displayImage(image):
        plt.imshow(image)
        plt.show()
'''