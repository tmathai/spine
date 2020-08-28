# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:50:36 2020

@author: 320065700
"""
'''
This class defines the iteration over the batches of images/masks. It handles all of the data transforms as well. 
'''

from multiprocessing import Process, Manager

import numpy as np
import re
from os import listdir
from os.path import isfile, join
import random

class BatchIterator(object):

    def __init__(
                self,
                epoch_transforms = [],
                iteration_transforms = [],

                input_batch_size = 16,
                input_iL_iteration_keys = None,

                ):

        self.epoch_transforms        = epoch_transforms
        self.iteration_transforms    = iteration_transforms

        # set batch size
        self.batch_size              = input_batch_size

        self.il_iteration_keys       = []
        # set iteration keys
        self.il_iteration_keys       = input_iL_iteration_keys

        self.data                    = {}

        self.shuffled_aug_keys       = []

        self.train_do_aug_flag       = True
        self.shuffle_datasets_flag   = True



    def __call__(self,
                 input_master_dict,

                 input_d_types_of_augs,

                 input_split_key = 'train',

                 input_shuffle_datasets_flag = True,

                 input_train_do_aug_flag = True
#                 iL_iteration_keys,
#                 input_types_of_augs_keys,
#                 motion_categories
                 ):

        # copy dictionary into input data
        self.data = input_master_dict.copy()

        # copy types of augs dict
        self.d_types_of_augs = input_d_types_of_augs.copy()

        self.input_split_key = input_split_key

        self.shuffle_datasets_flag = input_shuffle_datasets_flag

        self.train_do_aug_flag = input_train_do_aug_flag

        # if iteration keys not found, set it
        if not self.il_iteration_keys:
            self.il_iteration_keys = ['images', 'labels']

        # if shuffled aug keys not found, set it
        if not self.shuffled_aug_keys:
            # initially, these are not shuffled
            self.shuffled_aug_keys = list(self.d_types_of_augs.keys())



    def __iter__(self):

        # ========
        # epoch transforms

        print('before aug shuffle')
        print(self.shuffled_aug_keys)

        # initially, aug keys are not shuffled
        # get keys (convert to list), and shuffle them
        self.shuffled_aug_keys = random.sample(list(self.shuffled_aug_keys), len(self.shuffled_aug_keys))

        print("after aug shuffle")
        print(self.shuffled_aug_keys)

        # ========
        # shuffle filenames

        # shuffle flag
        if self.shuffle_datasets_flag == True:

            # print(self.data[self.input_split_key]['images'])
            # print(self.data[self.input_split_key]['labels'])

            c = list(zip(self.data[self.input_split_key]['images'] , self.data[self.input_split_key]['labels']))

            random.shuffle(c)

            self.data[self.input_split_key]['images'], self.data[self.input_split_key]['labels'] = zip(*c)

            # print(self.data[self.input_split_key]['images'])
            # print(self.data[self.input_split_key]['labels'])

        # ========
        # epoch transforms

#        for transform in self.epoch_transforms:
#
#            # apply shuffle transform to the augmentation keys [zoom, translate etc.]
#            self.shuffled_aug_keys = transform(self.types_of_augs_keys)

        # ========
        # iteration transforms

        # get images
        curr_dataset_images = self.data[self.input_split_key]['images']

        # get labels
        curr_dataset_labels = self.data[self.input_split_key]['labels']
#
        # number of images/labels in current dataset, for current category
        num_images = len(curr_dataset_images)
        num_labels = len(curr_dataset_labels)
        #print(num_images, num_labels)

        # check if num-images/labels are the same
        assert(num_images == num_labels)
#
        # compute batches
        batches = int(np.ceil(num_images / self.batch_size))

        #print('num_images:', num_images, 'num_labels:', num_labels, 'batches:', batches)

        # augmentation flag for training only
        if self.train_do_aug_flag == True:

            # iterate over data augmentation types (including no_aug)
            #for each_aug_idx, each_aug_val in enumerate(jdata["augmentations"]):
            for each_aug_name_key in self.shuffled_aug_keys:

                # iterate over batches
                for batch_idx in range(0,batches):

                    # get start and end index of batch
                    begin = batch_idx * self.batch_size
                    end   = begin + self.batch_size

                    # check if images are out of bounds
                    if end > num_images:
                        begin = num_images - self.batch_size
                        end = num_images

                    ffn = {}

                    # set augmentation key as current augmentation
                    ffn['aug_type'] = each_aug_name_key

                    # now set the other keys
                    for curr_iL_key in self.il_iteration_keys:
                        ffn[curr_iL_key] = []

                        if curr_iL_key == 'images':
                            ffn[curr_iL_key] = curr_dataset_images[begin:end]
                        if curr_iL_key == 'labels':
                            ffn[curr_iL_key] = curr_dataset_labels[begin:end]

                    # if batch_idx == 0:
                    #    print(each_aug_name_key)
                    #    print(ffn['images'])
                    #    print(ffn['labels'])

                    for idx, transform in enumerate(self.iteration_transforms):

                        # first iteration is reading images
                        if idx == 0:
                            batch = transform(ffn)

                        # last iteration is augmentations
                        elif idx == len(self.iteration_transforms) - 1:

                            # pass both batch of images and ffn['aug_type']
                            batch = transform(batch, ffn)

                        # all other iterations are data manipulation iterations
                        else:
                              batch = transform(batch)

                    yield batch

        else:

            # iterate over batches
            for batch_idx in range(0,batches):

                # get start and end index of batch
                begin = batch_idx * self.batch_size
                end   = begin + self.batch_size

                # check if images are out of bounds
                if end > num_images:
                    begin = num_images - self.batch_size
                    end = num_images

                ffn = {}

#                            # set augmentation key as current augmentation
#                            ffn['aug_type'] = each_aug_name_key

                # now set the other keys
                for curr_iL_key in self.il_iteration_keys:
                    ffn[curr_iL_key] = []

                    if curr_iL_key == 'images':
                        ffn[curr_iL_key] = curr_dataset_images[begin:end]
                    if curr_iL_key == 'labels':
                        ffn[curr_iL_key] = curr_dataset_labels[begin:end]

#                            if batch_idx == 0 and each_aug_name_key == 'no_aug':
#                                    print(ffn['images'])
#                                    print(ffn['labels'])

#                                if batch_idx == 0 :
#                                    print(each_aug_name_key)
#                                    print(ffn['images'])
#                                    print(ffn['labels'])

                for idx, transform in enumerate(self.iteration_transforms):

                    # first iteration is reading images
                    if idx == 0:
                        batch = transform(ffn)

                    # validation run has no augmentations
#                                # last iteration is augmentations
#                                elif idx == len(self.iteration_transforms) - 1:
#
#                                    # pass both batch of images and ffn['aug_type']
#                                    batch = transform(batch, ffn)

                    # all other iterations are data manipulation iterations
                    else:
                          batch = transform(batch)

                yield batch
