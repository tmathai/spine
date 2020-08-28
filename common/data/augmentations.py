# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:57:39 2020

@author: 320065700
"""
# os import
import sys
import os
from os import listdir
from os.path import isfile, join
import shutil
import errno

# math function import
from math import floor, ceil
from time import time
import timeit

# python import
import pandas as pd
import numpy as np
import scipy as sp
from scipy import misc
from matplotlib import pyplot as plt
#from sklearn.model_selection import KFold
import random
import copy
import re
import csv

# opencv import
import cv2

# import tifffile
import tifffile as tiff

# import nifti file handler
import nibabel as nib

import ast


## =================================================================================================
## =================================================================================================

def verticalFlipLeftRight_input_image(image, img_width, img_height):

    x = copy.deepcopy(np.asarray(image))
    xnp = x
#    xnp = np.asarray(x[:,:,0])
    xnpflip = np.fliplr(xnp)

    return xnpflip

## =================================================================================================
## =================================================================================================

def horizFlipUpDown_input_image(image, img_width, img_height):

    x = copy.deepcopy(np.asarray(image))
    xnp = x
#    xnp = np.asarray(x[:,:,0])
    xnpflip = np.flipud(xnp)

    return xnpflip

## =================================================================================================
## =================================================================================================

def bilateralFilter_input_image(image, img_width, img_height, ksize = 5, sigmaColor = 5, sigmaSpace = 5, iterations = 3):

    x = copy.deepcopy(np.asarray(image))
    xnp = x

    temp = copy.deepcopy(xnp)

    for itidx in range (0, iterations):

        bil = cv2.bilateralFilter(temp, ksize, sigmaColor, sigmaSpace)
        bil = np.asarray(bil).astype(np.float32)
        temp = copy.deepcopy(bil)


    return bil

## =================================================================================================
## =================================================================================================

def translate_input_image(image, img_width, img_height, translationValueX, translationValueY, imageHasBeenNormalizedFlag = True):

    x = copy.deepcopy(np.asarray(image))
    xnp = x

    f = np.zeros((img_height,img_width))

    # image has been normalized
    if imageHasBeenNormalizedFlag == True:
        d = copy.deepcopy(xnp)

        M = np.float32([[1,0,translationValueX],[0,1,translationValueY]])

        f = cv2.warpAffine(d, M, (img_width,img_height))

    # image has not been normalized
    if imageHasBeenNormalizedFlag == False:
        d = copy.deepcopy(xnp)

        M = np.float32([[1,0,translationValueX],[0,1,translationValueY]])

        f = cv2.warpAffine(d, M, (img_width,img_height)) # range [0,1] instead of [0,255]


    f = np.asarray(f).astype(np.float32)

    return f

## =================================================================================================
## =================================================================================================

def rotate_input_image(image, img_width, img_height, rotationAngleValue, imageHasBeenNormalizedFlag = True):

    x = copy.deepcopy(np.asarray(image))
    xnp = x

    f = np.zeros((img_height,img_width))

    # image has been normalized
    if imageHasBeenNormalizedFlag == True:
        d = copy.deepcopy(xnp)

        center = (img_width / 2, img_height / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, rotationAngleValue, scale)

        f = cv2.warpAffine(d, M, (img_height,img_width))

    # image has not been normalized
    if imageHasBeenNormalizedFlag == False:
        d = copy.deepcopy(xnp)

        center = (img_width / 2, img_height / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, rotationAngleValue, scale)

        f = cv2.warpAffine(d, M, (img_height,img_width)) # range [0,1] instead of [0,255]


    f = np.asarray(f).astype(np.float32)

    return f

## =================================================================================================
## =================================================================================================

def generate_flipLR(orig_image, orig_label):

    aug_img = copy.deepcopy(orig_image)

    img_height      = orig_image.shape[0]
    img_width       = orig_image.shape[1]
    img_channels    = orig_image.shape[2]

    aug_label = copy.deepcopy(orig_label)

    label_height      = orig_label.shape[0]
    label_width       = orig_label.shape[1]
    label_channels    = orig_label.shape[2]

    # -------
    # image augmentation

    for channel_idx in range(0,img_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_image[:, :, channel_idx])

        # flip
        aug_img[:, :, channel_idx] = verticalFlipLeftRight_input_image(curr_channel, img_width, img_height)

    # -------
    # label augmentation

    for channel_idx in range(0,label_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_label[:, :, channel_idx])

        # flip
        aug_label[:, :, channel_idx] = verticalFlipLeftRight_input_image(curr_channel, img_width, img_height)

    # -------
    # return img shape [ height, width, channels]
    # return label shape # return img shape [ height, width, channels = 1/2]
    return aug_img, aug_label

## =================================================================================================
## =================================================================================================

def generate_flipUD(orig_image, orig_label):

    aug_img = copy.deepcopy(orig_image)

    img_height      = orig_image.shape[0]
    img_width       = orig_image.shape[1]
    img_channels    = orig_image.shape[2]

    aug_label = copy.deepcopy(orig_label)

    label_height      = orig_label.shape[0]
    label_width       = orig_label.shape[1]
    label_channels    = orig_label.shape[2]

    # -------
    # image augmentation

    for channel_idx in range(0,img_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_image[:, :, channel_idx])

        # flip
        aug_img[:, :, channel_idx] = horizFlipUpDown_input_image(curr_channel, img_width, img_height)

    # -------
    # label augmentation

    for channel_idx in range(0,label_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_label[:, :, channel_idx])

        # flip
        aug_label[:, :, channel_idx] = horizFlipUpDown_input_image(curr_channel, img_width, img_height)

    # -------
    # return img shape [ height, width, channels]
    # return label shape # return img shape [ height, width, channels = 1/2]
    return aug_img, aug_label

## =================================================================================================
## =================================================================================================

def generate_bilateralBlur(orig_image, orig_label, aug_params, imageHasBeenNormalizedFlag = True):

    aug_img = copy.deepcopy(orig_image)

    img_height      = orig_image.shape[0]
    img_width       = orig_image.shape[1]
    img_channels    = orig_image.shape[2]

    aug_label = copy.deepcopy(orig_label)

    label_height      = orig_label.shape[0]
    label_width       = orig_label.shape[1]
    label_channels    = orig_label.shape[2]

    ksize = int(aug_params[0])
    sigmaColor = aug_params[1]
    sigmaSpace = aug_params[2]
    iterations = int(aug_params[3])

    # -------
    # image augmentation

    for channel_idx in range(0,img_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_image[:, :, channel_idx])

        # blur
        aug_img[:, :, channel_idx] = bilateralFilter_input_image(curr_channel, img_width, img_height, ksize = ksize, sigmaColor = sigmaColor, sigmaSpace = sigmaSpace, iterations = iterations)

    # -------
    # label augmentation

    for channel_idx in range(0,label_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_label[:, :, channel_idx])

        aug_label[:, :, channel_idx] = curr_channel
        # blur
        #aug_label[:, :, channel_idx] = bilateralFilter_input_image(curr_channel, img_width, img_height, ksize = ksize, sigmaColor = sigmaColor, sigmaSpace = sigmaSpace, iterations = iterations)

    # -------
    # return img shape [ height, width, channels]
    # return label shape # return img shape [ height, width, channels = 1/2]
    return aug_img, aug_label

## =================================================================================================
## =================================================================================================

def generate_translation(orig_image, orig_label, aug_params, imageHasBeenNormalizedFlag = True):

    aug_img = copy.deepcopy(orig_image)

    img_height      = orig_image.shape[0]
    img_width       = orig_image.shape[1]
    img_channels    = orig_image.shape[2]

    aug_label = copy.deepcopy(orig_label)

    label_height      = orig_label.shape[0]
    label_width       = orig_label.shape[1]
    label_channels    = orig_label.shape[2]

    translateX = int(aug_params[0])
    translateY = int(aug_params[1])

    # -------
    # image augmentation

    for channel_idx in range(0,img_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_image[:, :, channel_idx])

        # flip
        aug_img[:, :, channel_idx] = translate_input_image(curr_channel, img_width, img_height, translateX, translateY, imageHasBeenNormalizedFlag = True)

    # -------
    # label augmentation

    for channel_idx in range(0,label_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_label[:, :, channel_idx])

        # flip
        aug_label[:, :, channel_idx] = translate_input_image(curr_channel, img_width, img_height, translateX, translateY, imageHasBeenNormalizedFlag = True)

    # -------
    # return img shape [ height, width, channels]
    # return label shape # return img shape [ height, width, channels = 1/2]
    return aug_img, aug_label

## =================================================================================================
## =================================================================================================

def generate_rotation(orig_image, orig_label, aug_params, imageHasBeenNormalizedFlag = True):

    aug_img = copy.deepcopy(orig_image)

    img_height      = orig_image.shape[0]
    img_width       = orig_image.shape[1]
    img_channels    = orig_image.shape[2]

    aug_label = copy.deepcopy(orig_label)

    label_height      = orig_label.shape[0]
    label_width       = orig_label.shape[1]
    label_channels    = orig_label.shape[2]


    rotationAngle = int(random.randint(aug_params[0], aug_params[1]))

    # -------
    # image augmentation

    for channel_idx in range(0,img_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_image[:, :, channel_idx])

        # flip
        aug_img[:, :, channel_idx] = rotate_input_image(curr_channel, img_width, img_height, rotationAngle, imageHasBeenNormalizedFlag = True)

    # -------
    # label augmentation

    for channel_idx in range(0,label_channels):
        # shape is [height, width]
        curr_channel = copy.deepcopy(orig_label[:, :, channel_idx])

        # flip
        aug_label[:, :, channel_idx] = rotate_input_image(curr_channel, img_width, img_height, rotationAngle, imageHasBeenNormalizedFlag = True)

    # -------
    # return img shape [ height, width, channels]
    # return label shape # return img shape [ height, width, channels = 1/2]
    return aug_img, aug_label

## =================================================================================================
## =================================================================================================

def augment_batch(data, types_of_aug, curr_aug_type, num_elements):

    '''
    regular_images -- data['images'] size is [batch_size, height, width, channels]
    regular_labels -- data['labels'] size is [batch_size, height, width, channels = 1]
    one-hot_labels -- data['labels'] size is [batch_size, height, width, channels = 2]

    Example use case:
        1. If we load 5 input slices, images -- data['images'] size is [batch_size, height, width, channels = 5]
        2. Corresponding 5 output slices (normal), labels (regular) -- data['images'] size is [batch_size, height, width, channels = 1]
        3. Corresponding 5 output slices (one-hot), labels (one-hot) -- data['images'] size is [batch_size, height, width, channels = 2]
    '''

    # create new dict and create empty entries
    augmentations = {}
    for key in data.keys():
        augmentations[key] = []

    # get parameters for current aug type
    aug_params = ast.literal_eval(types_of_aug[curr_aug_type])

    aug_image_set = []
    aug_label_set = []

    for elem_idx in range(0,num_elements):

        if 'no_aug' in curr_aug_type:

            # do nothing, and return input dict
            aug_image_set = copy.deepcopy(data['images'])
            aug_label_set = copy.deepcopy(data['labels'])

        if 'flipped_LR' in curr_aug_type:

            orig_image = copy.deepcopy(data['images'][elem_idx])
            orig_label = copy.deepcopy(data['labels'][elem_idx])

            # returned img shape [[height, width, channels ]
            # returned label shape # return img shape [height, width, channels = 1/2]
            aug_img, aug_label = generate_flipLR(orig_image, orig_label)

            aug_image_set.append(aug_img)
            aug_label_set.append(aug_label)

        if 'flipped_UD' in curr_aug_type:

            orig_image = copy.deepcopy(data['images'][elem_idx])
            orig_label = copy.deepcopy(data['labels'][elem_idx])

            # returned img shape [[height, width, channels ]
            # returned label shape # return img shape [height, width, channels = 1/2]
            aug_img, aug_label = generate_flipUD(orig_image, orig_label)

            aug_image_set.append(aug_img)
            aug_label_set.append(aug_label)

        if 'bilateralBlur' in curr_aug_type:

            orig_image = copy.deepcopy(data['images'][elem_idx])
            orig_label = copy.deepcopy(data['labels'][elem_idx])

            # returned img shape [[height, width, channels ]
            # returned label shape # return img shape [height, width, channels = 1/2]
            aug_img, aug_label = generate_bilateralBlur(orig_image, orig_label, aug_params, imageHasBeenNormalizedFlag = True)

            aug_image_set.append(aug_img)
            aug_label_set.append(aug_label)

        if 'translate' in curr_aug_type:

            orig_image = copy.deepcopy(data['images'][elem_idx])
            orig_label = copy.deepcopy(data['labels'][elem_idx])

            # returned img shape [[height, width, channels ]
            # returned label shape # return img shape [height, width, channels = 1/2]
            aug_img, aug_label = generate_translation(orig_image, orig_label, aug_params, imageHasBeenNormalizedFlag = True)

            aug_image_set.append(aug_img)
            aug_label_set.append(aug_label)

        if 'rotate' in curr_aug_type:

            orig_image = copy.deepcopy(data['images'][elem_idx])
            orig_label = copy.deepcopy(data['labels'][elem_idx])

            # returned img shape [[height, width, channels ]
            # returned label shape # return img shape [height, width, channels = 1/2]
            aug_img, aug_label = generate_rotation(orig_image, orig_label, aug_params, imageHasBeenNormalizedFlag = True)

            aug_image_set.append(aug_img)
            aug_label_set.append(aug_label)

    # returns [batch_size, height, width, channels]
    augmentations['images'] = aug_image_set
    augmentations['labels'] = aug_label_set

    return augmentations

## =================================================================================================
## =================================================================================================

def perform_augmentations(types_of_augs = {}, keys = ['images', 'labels'],):
    '''
    This function takes as input a dictionary containing the various augmentation types (keys) and the parameters for each augmentation (values).
    It returns a function called transform.
    '''

    def transform(data, ffn):
        '''
        This function takes as input a dictionary, and uses the key ["images", "labels"] to access the images/labels.
        It performs the same type of augmentation on image/label, and returns the images/labels in a dictionary
        '''

        augmented = {}

        curr_aug_type = ffn['aug_type']
        #print(curr_aug_type)

        num_images = np.asarray(data['images']).shape[0]
        num_labels = np.asarray(data['labels']).shape[0]

        assert(num_images == num_labels)

        # get num_elems in batch if num_images and num_labels are the same
        num_elements = num_images

        augmented = augment_batch(data, types_of_augs, curr_aug_type, num_elements)

        return augmented

    return transform
