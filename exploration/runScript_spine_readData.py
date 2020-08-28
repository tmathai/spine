# os import
import sys
import os
from os import listdir
from os.path import isfile, join

# math function import
from math import floor, ceil
from time import time
import timeit

# python import
#import pandas as pd
import numpy as np
import scipy as sp
from scipy import misc, ndimage
#from matplotlib import pyplot as plt
#from sklearn.model_selection import KFold
import random
import copy

import re

import json
import ast

# tensorflow import
#import tensorflow as tf

# opencv import
import cv2

from matplotlib import pyplot as plt

import nibabel as nib

import argparse

## =================================================================================================
## =================================================================================================
# member functions

# sort all datasets in the folder based on the dataset #
def numericalSort(value):
    '''
    This function sorts the filenames in ascending order based on numbers in the filenames
    '''
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

## =================================================================================================
## =================================================================================================

def __read_all_sorted_image_files_in_directories(dummyListOfPaths):
    '''
    This function takes as input a list containing paths to a folder in all datasets. Then, for each dataset, it reads all image filenames, sorts them, and puts them into a container.
    '''
    # storage for file names inside folder
    base_ImageFileName_list = []
    fullPath_ImageFileName_list = []

    # for each path in list of paths
    for folderPath in dummyListOfPaths:

        base_imfn_list = []
        full_imfn_list = []

        # read each file in the folder
        base_ImageFileName_listTemp = [ f for f in listdir(folderPath) if isfile(join(folderPath,f)) ]

        # sort the file, and then store the sorted order of filenames
        for infile in sorted(base_ImageFileName_listTemp, key = numericalSort):
            # just filename
            base_imfn_list.append(infile)
            # filename and full path
            full_imfn_list.append(folderPath + infile)
            #print ("Current File Being Processed is: " + )

        base_ImageFileName_list.append(base_imfn_list)
        fullPath_ImageFileName_list.append(full_imfn_list)

    return base_ImageFileName_list, fullPath_ImageFileName_list



if __name__ == "__main__":

    rootFolder = 'C:/Users/320065700/OneDrive - Philips/Spine/Data/spine/'
    dataFolder = 'images/'
    maskFolder = 'masks/'

    bfn_images_dataset_list, ffp_images_dataset_list = __read_all_sorted_image_files_in_directories([rootFolder + dataFolder])

    bfn_masks_dataset_list, ffp_masks_dataset_list = __read_all_sorted_image_files_in_directories([rootFolder + maskFolder])

    print('num images', len(bfn_images_dataset_list[0]))
    print('num masks', len(bfn_masks_dataset_list[0]))

    #print('images')
    #print(ffp_images_dataset_list[0])
    #print('masks')
    #print(ffp_masks_dataset_list[0])

    assert len(bfn_images_dataset_list[0]) == len(bfn_masks_dataset_list[0])

    total_num_images = len(bfn_images_dataset_list[0])

    count_disc1_and_vert2 = 0
    count_disc1 = 0

    for img_idx in range(total_num_images):

        print('=====')
        print('idx', img_idx + 1)

        # read image
        ffn_image = ffp_images_dataset_list[0][img_idx]
        img = cv2.imread(ffn_image, cv2.IMREAD_UNCHANGED)

        print('ffn_image', ffn_image)
        print('shape', img.shape)
        print('data type', img.dtype)

        smallest = img.min()
        largest = img.max()
        print('img min', smallest, 'max', largest)

        # read mask
        ffn_mask = ffp_masks_dataset_list[0][img_idx]
        mask = np.load(ffn_mask)

        print('ffn_mask', ffn_mask)
        print('shape', mask.shape)
        print('data type', mask.dtype)


        smallest = mask.min()
        largest = mask.max()
        print('mask min', smallest, 'max', largest)

        assert img.shape == mask.shape

        if largest == 2:
            count_disc1_and_vert2 += 1

        if largest == 1:
            count_disc1 += 1

    print('count_disc1_and_vert2', count_disc1_and_vert2)
    print('count_disc1', count_disc1)
    print('count_other', total_num_images - count_disc1_and_vert2 - count_disc1)


    print('=====')

    img_idx = 14

    ffn_image = ffp_images_dataset_list[0][img_idx]
    print('ffn_image', ffn_image)
    img = cv2.imread(ffn_image, cv2.IMREAD_UNCHANGED)
    print('shape', img.shape)
    print('data type', img.dtype)

    ffn_mask = ffp_masks_dataset_list[0][img_idx]
    print('ffn_mask', ffn_mask)
    mask = np.load(ffn_mask)
    print('shape', mask.shape)
    print('data type', mask.dtype)

    smallest = mask.min()
    largest = mask.max()
    print(smallest, largest)


    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.show()
