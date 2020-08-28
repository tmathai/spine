# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:36:29 2019

@author: Tejas
"""

import os

import cv2
import numpy as np
import nibabel as nib
import copy


## =================================================================================================
## =================================================================================================


def read_original_input_images(images_paths):

    images = []

    for iidx, fn in enumerate(images_paths):

        # opencv laods input as BGR [H-768,W-1024,C-3] or grayscale [H-256,W-256]
        x = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

#        if iidx == 0:
#
#            print('id 0', iidx)
#            print('fn 0', fn)
#            print('x shape 0', x.shape, x.shape[0], x.shape[1])

        y = np.reshape(x, newshape=(x.shape[0], x.shape[1], 1))

        images.append(y)

    images = np.asarray(images)

    return images


## =================================================================================================
## =================================================================================================


def read_GT_npy(labels_paths):

    labels = []

    for fn in labels_paths:

        # opencv laods input as BGR [H-768,W-1024,C-3] or grayscale [H-256,W-256]
        x = np.load(fn)

        y = np.reshape(x, newshape=(x.shape[0], x.shape[1], 1))

        labels.append(y)

    labels = np.asarray(labels)

    return labels


## =================================================================================================
## =================================================================================================


def read_batch_images(keys = None):

    def transform(ffn):

        '''
        Given a container of full_file_name'ffn'  with two keys ['images', 'labels'], return a single container containing two keys:
            1. All the original input images loaded into a single key: 'images' - [numFrames_total, height, width, channels]
            2. All the GT label images loaded into a single key: 'labels' -  [numFrames_total, height, width, channels]
        '''
        data = {}

        if keys is not None:
            ffn_keys = keys
        else:
            ffn_keys = data.keys()

        for key in ffn_keys:
            data[key] = []

            if key == 'images':
                data[key] = read_original_input_images(ffn[key])
            else:
                data[key] = read_GT_npy(ffn[key])

        return data

    return transform
