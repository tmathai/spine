# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:04:43 2020

@author: 320065700
"""

'''
This class inherits the spineSegmentationNetwork and spineDataLoader classes. It is a simple wrapper.
'''

import numpy as np
import nibabel as nib

# project import
from spineSegNet import spineSegNet
from spineDataLoader import spineDataLoader


class spineSegmentation(spineSegNet, spineDataLoader):

    def __init__(self, jdata):


        print('Creating network')
        spineSegNet.__init__(self, jdata)

        print('Loading data')
        spineDataLoader.__init__(self, jdata)

        print('Completed network creation and data loading')
