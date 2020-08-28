# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:27:06 2019

@author: Tejas
"""

import numpy as np

class FitRunner(object):
    
    outputs = {}
    losses = {}
    placeholders = {}
    summaries = {}

    sess = None
    iteration = 0

    def __init__(self):
        self.create_placeholders()
        self.create_outputs()
        self.create_losses()
        #self.__create_summary_manager()
        self.create_training_op()
        
    def create_placeholders(self):
        self.placeholders = {}

    def create_outputs(self):
        self.outputs = {}

    def create_losses(self):
        self.losses = {}    

#    def __create_summary_manager(self):
#        raise NotImplementedError

    def create_training_op(self):
        raise NotImplementedError

    def run_epoch_train(self):
        raise NotImplementedError

    def run_epoch_valid(self):
        raise NotImplementedError

    def run_iteration(self, feed_dict, op_list, summaries):
        output_args = self.sess.run(op_list, feed_dict=feed_dict)
        return output_args