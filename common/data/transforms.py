# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:36:58 2019

@author: Tejas
"""
import numpy as np
import copy
import cv2
#from skimage.filters import gaussian
#from skimage.transform import resize




## =================================================================================================
## =================================================================================================

def convert_images_to_float(input_key = None):
    '''
    This function takes as input a key that identifies the input images. It returns a function called transform
    '''

    def transform(data):
        '''
        This function takes as input a dictionary, and uses the key "images" to access the images.
        It converts the images to float, and returns the images in a dictionary
        '''

        for key in input_key:

            if key == 'images':

                data[key] = np.asarray(data[key]).astype(np.float32)
                #print("\n data[key].shape", data[key].shape)

        return data

    return transform


## =================================================================================================
## =================================================================================================

def make_labels_onehot(num_classes):
    '''
    This function takes as input a scalar indicating the number of output classes. It returns a function called transform
    '''

    def transform(data):
        '''
        This function takes as input a dictionary, and uses the key "labels" to access the labels.
        It converts the labels to one-hot-encoding, and returns the labels in a dictionary
        '''

        labels_t = []

        for label in data['labels']:

            label_oneHot = np.zeros((label.shape[0], label.shape[1], num_classes), dtype=np.float32)

            for i in range(num_classes - 1, -1, -1):

                # vertebrae - 2
                # disc - 1
                # background - 0

                label_map = (label == i).astype(np.float32)
                label_map = label_map[:,:,0]
                label_oneHot[:, :, i] = copy.deepcopy(label_map)

                # # vertebrae
                # if i == 2:
                #     label_map = (label == 2).astype(np.float32)
                #     label_map = label_map[:,:,0]
                #     label_oneHot[:, :, 0] = copy.deepcopy(label_map)
                #
                # # disc
                # if i == 1:
                #     label_map = (label == 1).astype(np.float32)
                #     label_map = label_map[:,:,0]
                #     label_oneHot[:, :, 1] = copy.deepcopy(label_map)
                #
                # # background
                # if i == 0:
                #     label_map = (label == 0).astype(np.float32)
                #     label_map = label_map[:,:,0]
                #     label_oneHot[:, :, 2] = copy.deepcopy(label_map)






            labels_t.append(label_oneHot)

        data['labels'] = np.asarray(labels_t).astype(np.float32)

        return data

    return transform


## =================================================================================================
## =================================================================================================

def normalize_images_between_zero_and_one():
    '''
    This function takes as no input. It returns a function called transform.
    '''

    def transform(data):
        '''
        This function takes as input a dictionary, and uses the key "images" to access the images.
        It normalizes the images to [0,1], and returns the images in a dictionary
        '''

        images_t = []

        for image in data['images']:

            # normalize grayscale iamge
            if np.max(image) > 1.0:
                image /= 255.0

            images_t.append(image)

        data['images'] = np.stack(images_t)

        return data

    return transform


## =================================================================================================
## =================================================================================================

def normalize_labels_between_zero_and_one():
    '''
    This function takes as no input. It returns a function called transform.
    '''

    def transform(data):
        '''
        This function takes as input a dictionary, and uses the key "labels" to access the labels.
        It normalizes the labels to [0,1], and returns the labels in a dictionary
        '''

        labels_t = []

        for label in data['labels']:

            # normalize grayscale iamge
#            if np.max(label) > 1.0:

            label = np.asarray(label).astype(np.float32)

            label /= 255.0

            label[label > 0] = 1

            labels_t.append(label)

        data['labels'] = np.asarray(labels_t)

        return data

    return transform


## =================================================================================================
## =================================================================================================

def resize_data(keys = ['images', 'labels'], output_height = 256, output_width = 256, interp_method = cv2.INTER_LANCZOS4):
    '''
    This function takes as input the 'images'/'labels' keys. size is [batch_size, height, width, channels]
    It returns a function called transform.
    '''

    def transform(data):
        '''
        This function takes as input a dictionary, and uses the keys ["images", 'labels'] to access the images/labels.
        It resizes the images to [output_height,output_width], and returns the results in a dictionary
        '''


        resized_data = {}
        images_t = []
        labels_t = []

        for key in keys:

            if key == 'images':

                for image in data[key]:

                    img_height = image.shape[0]
                    img_width = image.shape[1]
                    img_channels = image.shape[2]

                    resized_image = np.zeros((output_height, output_width, img_channels), dtype=np.float32)

                    for channel_idx in range(0,img_channels):

                        curr_channel = np.asarray(image[:, :, channel_idx]).astype(np.float32)
                        resized_image[:,:,channel_idx] = cv2.resize(curr_channel, (output_width,output_height), interpolation = interp_method)

                    images_t.append(resized_image)


            if key == 'labels':

                for label in data[key]:

                    label_height = label.shape[0]
                    label_width = label.shape[1]
                    label_channels = label.shape[2]

                    resized_label = np.zeros((output_height, output_width, label_channels), dtype=np.float32)

                    for channel_idx in range(0,label_channels):

                        curr_channel = np.asarray(label[:, :, channel_idx]).astype(np.float32)
                        resized_channel = cv2.resize(curr_channel, (output_width,output_height), interpolation = interp_method)
                        resized_channel = (resized_channel > 0) * 255.0
                        resized_label[:,:,channel_idx] = resized_channel

                    labels_t.append(resized_label)


        resized_data['images'] = images_t
        resized_data['labels'] = labels_t

        # return [batch_size, height, width, channels]
        return resized_data

    return transform

## =================================================================================================
## =================================================================================================

def standardize_zeroMeanUnitVar_image(image, img_width, img_height, img_channels):

    x = copy.deepcopy(np.asarray(image))

    # mask region with only positive pixels
    binBrainRegion = x > 0

    # Get the pixel values in image where mask_value == 255, which may be later used to slice the array.
    img_mask = x[np.where(binBrainRegion == 1)]

    # get mean/std of masked region
    img_avg = np.mean(img_mask, axis=0)
    img_std = np.std(img_mask, axis=0)

    # convert sample data to zero mean
    img_zeroMean = (x - img_avg)

    # convert sample data to zero mean and unit variance
    img_unitVar = (img_zeroMean / img_std)

    # zero out background
    img_unitVar_masked = img_unitVar * binBrainRegion
    img_unitVar_masked[~binBrainRegion] = 0

    # convert to float32
    img_unitVar_masked = img_unitVar_masked.astype(np.float32)

    # send back masked brain that is zeroMean_unitVar standardized
    return img_unitVar_masked


def standardize_nifti_images_zeroMeanUnitVar(images_key = 'images'):
    '''
    This function takes as input the 'images' key. size is [batch_size, height, width, channels]
    It returns a function called transform.
    '''

    def transform(data):
        '''
        This function takes as input a dictionary, and uses the key "images" to access the images.
        It standardizes the images to 0 mean and unit variance, and returns the images in a dictionary
        '''

        images_t = []

        # data size is [batch_size, height, width, channels]

        for image in data[images_key]:

            img_height = image.shape[0]
            img_width = image.shape[1]
            img_channels = image.shape[2]

            # make temp copy of image
            standardized_image = copy.deepcopy(image)

            for channel_idx in range(0,img_channels):

                curr_channel_in_img = copy.deepcopy(image[:, :, channel_idx])

                # standardize image
                standardized_img = standardize_zeroMeanUnitVar_image(curr_channel_in_img, img_width, img_height, 1)

                # copy back into place
                standardized_image[:, :, channel_idx] = copy.deepcopy(standardized_img)

            images_t.append(standardized_image)

        data['images'] = np.asarray(images_t).astype(np.float32)

        # return [batch_size, height, width, channels]
        return data

    return transform


## =================================================================================================
## =================================================================================================

def normalize_nifti_labels_between_zero_and_one(labels_key = 'labels'):
    '''
    This function takes as input the 'labels' key. Size is [batch_size, height, width, channels = 1 or 3 or 5]
    It returns a function called transform.
    '''

    def transform(data):
        '''
        This function takes as input a dictionary, and uses the key "labels" to access the labels.
        It normalizes the labels to [0,1], and returns the labels in a dictionary
        '''

        labels_t = []

        # data Size is [batch_size, height, width, channels = 1 or 3 or 5]

        for label in data[labels_key]:

#            img_height = label.shape[0]
#            img_width = label.shape[1]
            img_channels = label.shape[2]

            # make temp copy of image
            normalized_label = copy.deepcopy(label)
            normalized_label = np.asarray(normalized_label).astype(np.float32)

            for channel_idx in range(0,img_channels):

                curr_channel_in_label = copy.deepcopy(label[:, :, channel_idx])

                curr_channel_in_label = np.asarray(curr_channel_in_label).astype(np.float32)

                curr_channel_in_label /= 255.0

                # copy back into place
                normalized_label[:, :, channel_idx] = copy.deepcopy(curr_channel_in_label)

            labels_t.append(normalized_label)

        data['labels'] = np.asarray(labels_t).astype(np.float32)

        # returns [batch_size, height, width, channels = 1 or 3 or 5]
        return data

    return transform


## =================================================================================================
## =================================================================================================

def make_nifti_labels_onehot(labels_key = 'labels', num_classes = 2):
    '''
    This function takes as inputs:
        1. a 'labels' key
        2. a scalar indicating the number of output classes.
    It returns a function called transform
    '''

    def transform(data):
        '''
        This function takes as input a dictionary, and uses the key "labels" to access the labels.
        It converts the labels to one-hot-encoding, and returns the labels in a dictionary
        '''

        labels_t = []

        # data['labels'] size is [batch_size, height, width, channels = 1]

        for label in data['labels']:

            label_oneHot = np.zeros((label.shape[0], label.shape[1], num_classes), dtype=np.float32)

            for i in range(num_classes):

                if i == 0:
                    # foreground
                    label_map = (label > 0).astype(np.float32)
                    label_map = label_map[:,:,0]
                    label_oneHot[:, :, i] = copy.deepcopy(label_map)
                else:
                    # background
                    label_map = (label == 0).astype(np.float32)
                    label_map = label_map[:,:,0]
                    label_oneHot[:, :, i] = copy.deepcopy(label_map)

            labels_t.append(label_oneHot)

        data['labels'] = np.asarray(labels_t).astype(np.float32)

        # data['labels'] input size was [batch_size, height, width, channels = 1]
        # returns [batch_size, height, width, channels = 2]
        return data

    return transform
