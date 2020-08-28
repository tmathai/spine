# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:17:24 2020

@author: 320065700
"""
'''
This class is the spine data loader class. It is used to load the input images and the labels. The data is then fed into the network as batches.
'''

import os
from os import listdir
from os.path import isfile, join

import random
from collections import namedtuple
import copy

import numpy as np

import pandas as pd

import json

import re
import cv2
import tensorflow as tf
import csv

import nibabel as nib

import ast

from sklearn.metrics import (
                                recall_score,
                                precision_score,
                                f1_score,
                                confusion_matrix
                            )

from utilities.iterators.batchiterators_spine import BatchIterator

from data.loading import (
                            read_batch_images,
                         )

from data.transforms import (
                                shuffle_filenames,
                                convert_images_to_float,
                                make_labels_onehot,
                                normalize_images_between_zero_and_one,
                                normalize_labels_between_zero_and_one,

                                resize_data,
                                standardize_nifti_images_zeroMeanUnitVar,
                                normalize_nifti_labels_between_zero_and_one,
                                make_nifti_labels_onehot,

                            )

from data.augmentations import (
                                perform_augmentations,
                                )

from utilities.tensorflow.saver import Saver

class spineDataLoader(object):

    ## =================================================================================================
    ## =================================================================================================
    ## =================================================================================================
    ## =================================================================================================
    # member variables



    num_total_train_images = 0
    num_total_valid_images = 0



    rootPath = ''
    rootFolder = ''
    storeOriginalFolder = ''
    storeAnnotationFolder = ''
    category_split_excel_fileName = ''
    image_label_assoc_excel_fileName = ''
    motion_categories = []
    contrast_types = []
    rootPathToRootFolder = ''


    choose_X_slices = 1
    choose_X_GT_slices = 1
    # default value for BRATS
    num_vol_slices = 155
    batch_size = 8
    iteration_keys = ['images', 'labels']

    # dummy
    network_input_image_width = 256
    network_input_image_height = 256
    network_input_channels = 1
    network_output_channels = 1


    types_of_augs = {}


    active_loss = None


    graph = {}
    sess = {}

    ## =================================================================================================
    ## =================================================================================================
    ## =================================================================================================
    ## =================================================================================================
    # member functions


    def __init__(self, jdata):
        '''
        This function is the class constructor (not default).
        '''

        # =====================================================================
        # =====================================================================
        # data loading

        # initialize folder names
        self.rootPath = jdata['data_loading']['rootPath']
        self.rootFolder = jdata['data_loading']['rootFolder']
        #self.trainFolder = trainFolder
        #self.validFolder = validFolder
        self.storeOriginalFolder = jdata['data_loading']['storeOriginalFolder']
        self.storeAnnotationFolder = jdata['data_loading']['storeAnnotationFolder']

        # get names of excel files
        self.category_split_excel_fileName = jdata['data_loading']['rootFolder'].split('/')[0] + jdata['data_loading']['category_split_excel_fileName']
        self.image_label_assoc_excel_fileName = jdata['data_loading']['image_label_assoc_excel_fileName']

        # get motion categories
        self.motion_categories = ast.literal_eval(jdata['data_loading']['motion_categories'])

        # get MRI contrast types
        self.contrast_types = ast.literal_eval(jdata['data_loading']['contrast_types'])

        # initialize paths
        self.rootPathToRootFolder = self.rootPath + '/' + self.rootFolder + '/'
        #self.fullPathToTrainFolder = self.rootPathToRootFolder + trainFolder + '/'
        #self.fullPathToValidFolder = self.rootPathToRootFolder + validFolder + '/'

        # =====================================================================
        # =====================================================================
        # data params

        self.original_input_image_width = jdata['data_params']['original_input_image_width']

        self.original_input_image_height = jdata['data_params']['original_input_image_height']

        self.network_input_image_width = jdata['data_params']['network_input_image_width']

        self.network_input_image_height = jdata['data_params']['network_input_image_height']

        self.choose_X_slices = jdata['data_params']['choose_X_slices']

        self.choose_X_GT_slices = jdata['data_params']['choose_X_GT_slices']

        self.num_vol_slices = jdata['data_params']['num_vol_slices']

        self.iteration_keys = ast.literal_eval(jdata['data_params']['iteration_keys'])

        # =====================================================================
        # =====================================================================
        # training params

        self.batch_size = jdata['training_params']['batch_size']

        self.dropout_training_Flag = jdata['training_params']['dropout_training_Flag']

        self.dropout_prob_training = jdata['training_params']['dropout_prob_training']

        self.dropout_prob_testing = jdata['training_params']['dropout_prob_testing']

        # =====================================================================
        # =====================================================================
        # learning type

        self.learn_type = jdata['learning_type']['learn_type']

        # =====================================================================
        # =====================================================================
        # model output params

        # number of input channels equals the number of slices to select
        self.network_input_channels = self.choose_X_slices

        self.network_output_channels = jdata['model_output_params']['network_output_channels']

        # =====================================================================
        # =====================================================================
        # data augmentations

        self.types_of_augs = jdata['data_augmentations']

        self.train_shuffle_datasets_flag = True
        self.valid_shuffle_datasets_flag = True

        self.train_do_aug_flag = True
        self.valid_do_aug_flag = True

#        "train_shuffle_datasets_flag": True
#        "train_do_aug_flag": True
#        "valid_shuffle_datasets_flag": True
#        "valid_do_aug_flag": = False

        # =====================================================================
        # =====================================================================
        # algorithm



        # =====================================================================
        # =====================================================================
        #

        # step 1 - get filenames
        # read all the images/masks filenames
        self.__create_dict_images_and_masks(jdata)

        # step 2 - prune images/labels
        # ATTENTION!
        # small dataset, read the images/masks, and prune files based on labels
        # must contain disc (1), vertebrae (2) and background (0)
        self.__prune_dict_images_and_masks()

        # step 3 - get data split
        # get train/valid/test split
        self.__create_dict_trainValidTestSplit(train = 80, valid = 20)

        # step 4 - create iterator over training data
        self.__create_train_batch_iterator()

        # step 5 - create iterator over validation data
        self.__create_valid_batch_iterator()

        # step 6 - create TF graphs and sessions
        self.__create_tf_graph_session()

        # step 7 - create summary manager for results
        self.__create_summary_manager()

        # step 8 - get number of total training batches (including augmentation)
        self.num_total_batches_train = self.__compute_total_batches(
                                                                        self.ffn_master_dict_spine_images_and_labels,
                                                                        input_split_key = 'train',
                                                                        do_aug_flag = True,
                                                                    )

        # step 9 - get number of total validation batches (including augmentation)
        self.num_total_batches_valid = self.__compute_total_batches(
                                                                        self.ffn_master_dict_spine_images_and_labels,
                                                                        input_split_key = 'valid',
                                                                        do_aug_flag = True,
                                                                    )


    ## =================================================================================================
    ## =================================================================================================
    # member functions

    # sort all datasets in the folder based on the dataset #
    def numericalSort(self, value):
        '''
        This function sorts the filenames in ascending order based on numbers in the filenames
        '''
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    ## =================================================================================================
    ## =================================================================================================

    def __read_all_sorted_image_files_in_directories(self, dummyListOfPaths):
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
            for infile in sorted(base_ImageFileName_listTemp, key = self.numericalSort):
                # just filename
                base_imfn_list.append(infile)
                # filename and full path
                full_imfn_list.append(folderPath + infile)
                #print ("Current File Being Processed is: " + )

            base_ImageFileName_list.append(base_imfn_list)
            fullPath_ImageFileName_list.append(full_imfn_list)

        return base_ImageFileName_list, fullPath_ImageFileName_list

    ## =================================================================================================
    ## =================================================================================================

    def __create_masterDict_assoc_images_and_masks(
                                                    self,
                                                    rootPathToRootFolder,
                                                    imageFolder = 'images/',
                                                    labelsFolder = 'masks/',
                                                    image_ext = '.png',
                                                    mask_ext = '.npy'
                                                    ):

        '''
        This function creates a master dictionary with all the image filenames, and associated labels filenames.
        '''

        # full file path to label and associated image file
        ffn_master_dict_images_and_labels = {}

        # only the number
        # 81
        bfnumber_master_dict_images_and_labels = {}

        # read all the images in the folder
        bfn_images_dataset_list, ffp_images_dataset_list = self.__read_all_sorted_image_files_in_directories([rootPathToRootFolder + imageFolder])

        # read all the masks in the folder
        bfn_masks_dataset_list, ffp_masks_dataset_list = self.__read_all_sorted_image_files_in_directories([rootPathToRootFolder + labelsFolder])

        # store images (get the first element from list, which is also a list)
        bfnumber_master_dict_images_and_labels['images_all'] = bfn_images_dataset_list[0]
        ffn_master_dict_images_and_labels['images_all'] = ffp_images_dataset_list[0]

        # store masks (get the first element from list, which is also a list)
        bfnumber_master_dict_images_and_labels['labels_all'] = bfn_masks_dataset_list[0]
        ffn_master_dict_images_and_labels['labels_all'] = ffp_masks_dataset_list[0]

        # print('num images', len(bfn_images_dataset_list[0]))
        # print('num masks', len(bfn_masks_dataset_list[0]))
        #
        # print('images')
        # print(ffp_images_dataset_list[0])
        # print('masks')
        # print(ffp_masks_dataset_list[0])

        return ffn_master_dict_images_and_labels, bfnumber_master_dict_images_and_labels

    ## =================================================================================================
    ## =================================================================================================

    def __create_dict_images_and_masks(self, jdata):
        '''
        This function is a simple wrapper to create master dictionary with images/masks filenames.
        '''

        # OUTPUT
        # master_dict_images_masks['images' or 'masks'][N]
        # 'images' stands for the grayscale images_idx
        # 'masks' stands for the masks (background 0, disc 1, vertebrae 2)
        # [N] stands for the total number of images/masks for this dataset

        self.ffn_master_dict_spine_images_and_labels, \
        self.bfnumber_master_dict_spine_images_and_labels,        = self.__create_masterDict_assoc_images_and_masks(
                                                                                                                                    self.rootPathToRootFolder,
                                                                                                                                    imageFolder = self.storeOriginalFolder,
                                                                                                                                    labelsFolder = self.storeAnnotationFolder,
                                                                                                                                    image_ext = '.png',
                                                                                                                                    mask_ext = '.npy'
                                                                                                                                    )


    ## =================================================================================================
    ## =================================================================================================

    def __prune_dict_images_and_masks(self):
        '''
        This function reads the images/masks from filenames, checks the labels (include all 0,1,2 labels only), and discards others
        '''

        # create store for pruned images
        self.ffn_master_dict_spine_images_and_labels['images_pruned'] = []
        self.ffn_master_dict_spine_images_and_labels['labels_pruned'] = []

        self.bfnumber_master_dict_spine_images_and_labels['images_pruned'] = []
        self.bfnumber_master_dict_spine_images_and_labels['labels_pruned'] = []

        # get all images
        ffn_images = self.ffn_master_dict_spine_images_and_labels['images_all']
        ffn_labels = self.ffn_master_dict_spine_images_and_labels['labels_all']

        bfn_images = self.bfnumber_master_dict_spine_images_and_labels['images_all']
        bfn_labels = self.bfnumber_master_dict_spine_images_and_labels['labels_all']

        assert len(ffn_images) == len(ffn_labels)

        total_num_images = len(ffn_images)
        print('original total_num_images', total_num_images)

        count_disc1_and_vert2 = 0

        for img_idx in range(total_num_images):

            # print('=====')
            # print('idx', img_idx + 1)

            # read image
            ffn_image = ffn_images[img_idx]
            img = cv2.imread(ffn_image, cv2.IMREAD_UNCHANGED)

            # print('ffn_image', ffn_image)
            # print('shape', img.shape)
            # print('data type', img.dtype)

            smallest = img.min()
            largest = img.max()
            # print('img min', smallest, 'max', largest)

            # read mask
            ffn_mask = ffn_labels[img_idx]
            mask = np.load(ffn_mask)

            # print('ffn_mask', ffn_mask)
            # print('shape', mask.shape)
            # print('data type', mask.dtype)


            smallest = mask.min()
            largest = mask.max()
            # print('mask min', smallest, 'max', largest)

            assert img.shape == mask.shape

            if largest == 2:
                count_disc1_and_vert2 += 1
                self.ffn_master_dict_spine_images_and_labels['images_pruned'].append(ffn_image)
                self.ffn_master_dict_spine_images_and_labels['labels_pruned'].append(ffn_mask)

                self.bfnumber_master_dict_spine_images_and_labels['images_pruned'].append(bfn_images[img_idx])
                self.bfnumber_master_dict_spine_images_and_labels['labels_pruned'].append(bfn_labels[img_idx])


        # get all images
        ffn_images_pruned = self.ffn_master_dict_spine_images_and_labels['images_pruned']
        ffn_labels_pruned = self.ffn_master_dict_spine_images_and_labels['labels_pruned']

        assert len(ffn_images_pruned) == len(ffn_labels_pruned)

        total_num_images = len(ffn_images_pruned)
        print('pruned total_num_images', total_num_images)

    ## =================================================================================================
    ## =================================================================================================

    def __create_dict_trainValidTestSplit(self, train = 80, valid = 20):
        '''
        This function splits the pruned data into train/valid/test splits
        '''

        # get pruned images
        ffn_images_pruned = self.ffn_master_dict_spine_images_and_labels['images_pruned']
        ffn_labels_pruned = self.ffn_master_dict_spine_images_and_labels['labels_pruned']

        # get pruned images
        bfn_images_pruned = self.bfnumber_master_dict_spine_images_and_labels['images_pruned']
        bfn_labels_pruned = self.bfnumber_master_dict_spine_images_and_labels['labels_pruned']

        assert len(ffn_images_pruned) == len(ffn_labels_pruned)

        total_num_images = len(ffn_images_pruned)
        print('pruned total_num_images', total_num_images)

        dummy_range = np.arange(total_num_images)
        print('dummy_range', dummy_range)

        num_train = int(np.floor((train * total_num_images)/100))
        print('num_train', num_train)

        num_valid = int(np.floor((valid * total_num_images)/100))
        print('num_valid', num_valid)

        # data = np.random.rand(15, 1)
        # v = data[np.random.choice(len(data), size=3, replace=False)]
        # #v = random.sample(data, 3)
        # print(v)

        train_idxs = dummy_range[np.random.choice(len(dummy_range), size=num_train, replace=False)]

        #train_idxs = [i for i in dummy_range if i in train_idxs]
        train_idxs = np.sort(train_idxs)

        valid_idxs = np.setdiff1d(dummy_range, train_idxs)

        print('train_idxs', train_idxs)
        print('valid_idxs', valid_idxs)

        assert num_train == len(train_idxs)
        assert num_valid == len(valid_idxs)

        # ====================
        # ====================

        # create train dataset
        self.ffn_master_dict_spine_images_and_labels['train'] = {}
        self.ffn_master_dict_spine_images_and_labels['train']['images'] = []
        self.ffn_master_dict_spine_images_and_labels['train']['labels'] = []

        self.bfnumber_master_dict_spine_images_and_labels['train'] = {}
        self.bfnumber_master_dict_spine_images_and_labels['train']['images'] = []
        self.bfnumber_master_dict_spine_images_and_labels['train']['labels'] = []

        for tr_idx in train_idxs:
            self.ffn_master_dict_spine_images_and_labels['train']['images'].append(ffn_images_pruned[tr_idx])
            self.ffn_master_dict_spine_images_and_labels['train']['labels'].append(ffn_labels_pruned[tr_idx])

            self.bfnumber_master_dict_spine_images_and_labels['train']['images'].append(bfn_images_pruned[tr_idx])
            self.bfnumber_master_dict_spine_images_and_labels['train']['labels'].append(bfn_labels_pruned[tr_idx])

        print('train_images_idxs', self.bfnumber_master_dict_spine_images_and_labels['train']['images'])
        print('train_label_idxs', self.bfnumber_master_dict_spine_images_and_labels['train']['labels'])

        # ====================
        # ====================

        # create valid dataset
        self.ffn_master_dict_spine_images_and_labels['valid'] = {}
        self.ffn_master_dict_spine_images_and_labels['valid']['images'] = []
        self.ffn_master_dict_spine_images_and_labels['valid']['labels'] = []

        self.bfnumber_master_dict_spine_images_and_labels['valid'] = {}
        self.bfnumber_master_dict_spine_images_and_labels['valid']['images'] = []
        self.bfnumber_master_dict_spine_images_and_labels['valid']['labels'] = []

        for v_idx in valid_idxs:
            self.ffn_master_dict_spine_images_and_labels['valid']['images'].append(ffn_images_pruned[v_idx])
            self.ffn_master_dict_spine_images_and_labels['valid']['labels'].append(ffn_labels_pruned[v_idx])

            self.bfnumber_master_dict_spine_images_and_labels['valid']['images'].append(bfn_images_pruned[v_idx])
            self.bfnumber_master_dict_spine_images_and_labels['valid']['labels'].append(bfn_labels_pruned[v_idx])

        print('valid_images_idxs', self.bfnumber_master_dict_spine_images_and_labels['valid']['images'])
        print('valid_label_idxs', self.bfnumber_master_dict_spine_images_and_labels['valid']['labels'])



    ## =================================================================================================
    ## =================================================================================================



    def __create_train_batch_iterator(self):
        '''
            This function creates a iterator across the batches of images in the training datasets
        '''

        if self.network_output_channels == 1:

            self.batch_iterator_train = BatchIterator(

                                                            epoch_transforms = [],

                                                            iteration_transforms = [
                                                                                    read_batch_images(keys = ['images', 'labels']),

                                                                                    convert_images_to_float(input_key = ['images']),

                                                                                    normalize_images_between_zero_and_one(),

                                                                                    normalize_labels_between_zero_and_one(),

                                                                                    perform_augmentations(
                                                                                                            types_of_augs = self.types_of_augs,
                                                                                                            keys = ['images', 'labels'],
                                                                                                            ),
                                                                                    ],

                                                            input_batch_size = self.batch_size,

                                                            input_iL_iteration_keys = self.iteration_keys



                                                        )
        else:

            self.batch_iterator_train = BatchIterator(

                                                            epoch_transforms = [],

                                                            iteration_transforms = [
                                                                                    read_batch_images(keys = ['images', 'labels']),

                                                                                    convert_images_to_float(input_key = ['images']),

                                                                                    normalize_images_between_zero_and_one(),

                                                                                    make_labels_onehot(num_classes = self.network_output_channels),

                                                                                    perform_augmentations(
                                                                                                            types_of_augs = self.types_of_augs,
                                                                                                            keys = ['images', 'labels'],
                                                                                                            ),
                                                                                    ],

                                                            input_batch_size = self.batch_size,

                                                            input_iL_iteration_keys = self.iteration_keys



                                                        )


        self.batch_iterator_train(
                                    input_master_dict = self.ffn_master_dict_spine_images_and_labels,

                                    input_d_types_of_augs = self.types_of_augs.copy(),

                                    input_split_key = 'train',

                                    input_shuffle_datasets_flag = self.train_shuffle_datasets_flag,

                                    input_train_do_aug_flag = self.train_do_aug_flag
                                    )



    ## =================================================================================================
    ## =================================================================================================


    def __create_valid_batch_iterator(self):
        '''
            This function creates a iterator across the batches of images in the validation datasets
        '''

        if self.network_output_channels == 1:

            self.batch_iterator_valid = BatchIterator(

                                                            epoch_transforms = [],

                                                            iteration_transforms = [
                                                                                    read_batch_images(keys = ['images', 'labels']),

                                                                                    convert_images_to_float(input_key = ['images']),

                                                                                    normalize_images_between_zero_and_one(),

                                                                                    normalize_labels_between_zero_and_one(),

                                                                                    perform_augmentations(
                                                                                                            types_of_augs = self.types_of_augs,
                                                                                                            keys = ['images', 'labels'],
                                                                                                            ),
                                                                                    ],

                                                            input_batch_size = self.batch_size,

                                                            input_iL_iteration_keys = self.iteration_keys



                                                        )
        else:

            self.batch_iterator_valid = BatchIterator(

                                                            epoch_transforms = [],

                                                            iteration_transforms = [
                                                                                    read_batch_images(keys = ['images', 'labels']),

                                                                                    convert_images_to_float(input_key = ['images']),

                                                                                    normalize_images_between_zero_and_one(),

                                                                                    make_labels_onehot(num_classes = self.network_output_channels),

                                                                                    perform_augmentations(
                                                                                                            types_of_augs = self.types_of_augs,
                                                                                                            keys = ['images', 'labels'],
                                                                                                            ),
                                                                                    ],

                                                            input_batch_size = self.batch_size,

                                                            input_iL_iteration_keys = self.iteration_keys



                                                        )


        self.batch_iterator_valid(
                                    input_master_dict = self.ffn_master_dict_spine_images_and_labels,

                                    input_d_types_of_augs = self.types_of_augs.copy(),

                                    input_split_key = 'valid',

                                    input_shuffle_datasets_flag = self.valid_shuffle_datasets_flag,

                                    input_train_do_aug_flag = self.valid_do_aug_flag
                                    )


    ## =================================================================================================
    ## =================================================================================================


    def __create_tf_graph_session(self):
        '''
            Create tensorflow variables, graphs and session info
        '''

        self.graph = tf.Graph()
        self.sess = tf.Session()

        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)


    ## =================================================================================================
    ## =================================================================================================

    def __create_summary_manager(self):
        '''
            Create tensorflow model saver
        '''

        self.saver = Saver(
                                self.sess,
                                self.placeholders,
                                self.outputs,
                                self.savepath
                            )

    ## =================================================================================================
    ## =================================================================================================

    def __compute_total_batches(self, data, input_split_key = 'train', do_aug_flag = True):
        '''
            Get total number of batches (including augmentation)
        '''

        #print('do_aug_flag', do_aug_flag)

        # aug_types
        num_aug_types = len(list(self.types_of_augs.keys()))

        total_number_of_batches = 0

        # get images
        curr_dataset_images = data[input_split_key]['images']

        # get labels
        curr_dataset_labels = data[input_split_key]['labels']
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

        if do_aug_flag == True:
            total_number_of_batches = total_number_of_batches + (batches * num_aug_types)
        else:
            total_number_of_batches += batches

        return total_number_of_batches
