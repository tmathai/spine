# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 23:25:09 2018

@author: Tejas
"""

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

import json
import ast

# project import
from spineSegmentation import spineSegmentation

# tensorflow import
import tensorflow as tf

# opencv import
import cv2

from matplotlib import pyplot as plt

import nibabel as nib

import argparse


if __name__ == "__main__":


    trainFlag = True

    testFlag = False

    hypercolumnFlag = False



    # =============
    # enter json file on command line

    parser = argparse.ArgumentParser("Training a neural network with a JSON file")
    parser.add_argument("json_file", help="JSON parameters file -- Enter the path + filename", type=str)
    args = parser.parse_args()

    # get json file
    json_file = args.json_file

    # =============
    # load the json file

    jdata = json.load(open(json_file))

    # =============
    # read epochs (training iterations) for training

    # get the number of epochs over training data
    n_epochs = jdata['training_params']['n_epochs']


    ## =================================================================================================
    ## =================================================================================================

    if trainFlag == True:

        # create class object
        fit_runner = spineSegmentation(jdata)

        # epochs for early stopping of training (prevent overfitting)
        patience_iterations = 7

        # training process over epochs
        for epoch in range(n_epochs):

            print('\n')
            print('==========')
            print('epoch {}'.format(epoch + 1))
            print('==========')
            print('\n')

            fit_runner.run_epoch_train(epoch)

            # counter for number of epochs with no decrease in validation loss
            counter_no_dec_in_val_loss = fit_runner.run_epoch_valid(epoch)

            # early stop if there is no improvement in val_loss
            if counter_no_dec_in_val_loss == patience_iterations:
                break

        # save summary of training process
        fit_runner.save_training_summaries()
