# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:32:57 2018

@author: Tejas
"""
'''
This class defines the segmentation network for the batches of images/masks. It handles the placeholders, losses, and training results summary.
'''

import os
import sys

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import tensorflow as tf

from utilities.fitting.fitrunners import FitRunner
from utilities.timing import timeit
from layers.losses import dice, dice_loss, binary_cross_entropy_2D

from models import (
                        # no dropout - UNet
                        unet_reg_v1,

                        # dropout - UNet
                        unet_reg_v1_dropout,
                    )



class SegBaseclass(FitRunner):



    ## =================================================================================================
    ## =================================================================================================

    def initialize_random_generators(self):
        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)

    ## =================================================================================================
    ## =================================================================================================

    def create_placeholders(self):

        raise NotImplementedError('create_placeholders is not defined in this class')

    ## =================================================================================================
    ## =================================================================================================

    def create_outputs(self):

        raise NotImplementedError('Create_outputs is not defined in this class')

    ## =================================================================================================
    ## =================================================================================================

    def my_tf_round(self, x, decimals = 0):

        multiplier = tf.constant(10**decimals, dtype=x.dtype)

        return tf.round(x * multiplier) / multiplier

    ## =================================================================================================
    ## =================================================================================================

    def create_losses(self):

        raise NotImplementedError('Create_losses is not defined in this class')

    ## =================================================================================================
    ## =================================================================================================

    def create_training_op(self):

        raise NotImplementedError('The training operation is not defined in this class')

    ## =================================================================================================
    ## =================================================================================================

    @timeit
    def run_epoch_train(self):
        raise NotImplementedError('The training loop is not defined in this class')

    ## =================================================================================================
    ## =================================================================================================

    @timeit
    def run_epoch_valid(self):
        raise NotImplementedError('The validation loop is not defined in this class')

    ## =================================================================================================
    ## =================================================================================================

#    def __create_summary_manager(self):
#        raise NotImplementedError('The function is not defined in this class')
#
#    def create_summaries(self):
#
#        self.summaries = {'all': tf.summary.merge_all()}

    ## =================================================================================================
    ## =================================================================================================


class spineSegNet(SegBaseclass):

    # create dummy variables
    savepath = ''

    summary_path = ''

    dataset_name = ''

    learning_rate =  0.00001

    iteration = 0

    batch_size = 16

    network_output_channels = 2

    overall_best_score = sys.float_info.max

    decimal_truncate_limit = 7

    active_loss = None

    per_epoch_summary_data = {
                                'epoch': [],
                                'train_loss': [],
                                'train_dice': [],
                                'val_loss': [],
                                'val_dice': []

                            }


    def __init__(self, jdata):

        # =====================================================================
        # =====================================================================
        # algorithm params

        self.network_name = jdata['algorithm']['name']

        self.learn_type = jdata['learning_type']['learn_type']

        self.active_loss = jdata['algorithm']['loss']

        self.run_name = jdata['algorithm']['run_name']

        self.net_type = jdata['algorithm']['net_type']

        # =====================================================================
        # =====================================================================
        # training params

        self.batch_size = jdata['training_params']['batch_size']

        self.n_epochs = jdata['training_params']['n_epochs']

        self.learning_rate = jdata['training_params']['learning_rate']

        self.dropout_training_Flag = jdata['training_params']['dropout_training_Flag']

        self.dropout_prob_training = jdata['training_params']['dropout_prob_training']

        self.dropout_prob_testing = jdata['training_params']['dropout_prob_testing']

        # =====================================================================
        # =====================================================================
        # dataset params

        self.dataset_name = jdata['dataset']['name']

        # =====================================================================
        # =====================================================================
        # data params

        self.original_input_image_width = jdata['data_params']['original_input_image_width']

        self.original_input_image_height = jdata['data_params']['original_input_image_height']

        self.network_input_image_width = jdata['data_params']['network_input_image_width']

        self.network_input_image_height = jdata['data_params']['network_input_image_height']

        # number of input channels equals the number of slices to select
        self.network_input_channels = jdata['data_params']['choose_X_slices']

        self.network_input_size = [self.network_input_image_height, self.network_input_image_width]

        # =====================================================================
        # =====================================================================
        # model output params

        self.network_output_channels = jdata['model_output_params']['network_output_channels']

        self.model_folder = jdata['model_output_params']['modelFolderPath']

        # =====================================================================
        # =====================================================================
        # data loading

        self.type_of_data = jdata['data_loading']['type_of_data']


        self.counter_no_dec_in_val_loss = 0

#        batch_size = jdata['batch_size']
#        self.batch_size = batch_size
#
#        dataset_name = jdata['dataset_name']
#        self.dataset_name = dataset_name # 'BOE_2018_dropout'
#
#        self.network_name = jdata['network_name'] # '_LSTMResNet_'



        self.create_save_folders()


        super(spineSegNet, self).__init__()


    ## =================================================================================================
    ## =================================================================================================

    def create_save_folders(self):

        # save model in this folder
        self.savepath = self.model_folder + '/' + self.network_name + '_' + str(self.net_type) + '_' + self.learn_type + '_' + self.type_of_data + '_' + 'b' + str(self.batch_size) + '_' + 'e' + str(self.n_epochs) + '_' + self.active_loss + '_' + self.run_name

        # create folder
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        # save summary of training in this folder
        self.summary_path = self.savepath + '/summaries'

        # create folder
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)


    ## =================================================================================================
    ## =================================================================================================

    def create_placeholders(self):

        self.placeholders['images'] = tf.placeholder(
                                                        shape=[None] + self.network_input_size + [self.network_input_channels],
                                                        name='images',
                                                        dtype=tf.float32
                                                    )

        self.placeholders['labels'] = tf.placeholder(
                                                        shape=[None] + self.network_input_size + [self.network_output_channels],
                                                        name='labels',
                                                        dtype=tf.float32
                                                    )

        self.placeholders['is_training'] = tf.placeholder(
                                                            shape=[],
                                                            name='is_training',
                                                            dtype=tf.bool
                                                        )

        # dropout is included
        if self.dropout_training_Flag == 1:

            self.placeholders['dropout_prob'] = tf.placeholder(
                                                                shape=[],
                                                                name='dropout_prob',
                                                                dtype=tf.float32
                                                            )


    ## =================================================================================================
    ## =================================================================================================

    def create_outputs(self):

        # get network
        if self.network_name == 'UNet_reg':

            print('\n')
            print('Training UNet_reg')

            # dropout is included
            if self.dropout_training_Flag == 1:

                # train with dropout
                output_from_network = unet_reg_v1_dropout(
                                            self.placeholders['images'], self.placeholders['is_training'], self.placeholders['dropout_prob'], self.net_type, self.network_output_channels
                                        )

            else:

                # train without dropout
                output_from_network = unet_reg_v1(
                                            self.placeholders['images'], self.placeholders['is_training'], self.net_type, self.network_output_channels
                                        )

        # store model output
        self.outputs['logits'] = output_from_network

        # if one channel, then store/segment
        if self.network_output_channels == 1:

            self.final_output = output_from_network
            self.outputs['sigmoid'] = output_from_network
            self.outputs['segmentation'] = self.outputs['sigmoid'] > 0.5

        # multi-channel, then store/segment
        else:

            self.final_output = output_from_network
            self.outputs['sigmoid'] = output_from_network
            self.outputs['segmentation'] = self.outputs['sigmoid'] > 0.5

#        # not a dice loss, just dice coefficient
#        ## input labels, prediction --> [None, height, width, num_classes]
#        ## output --> [None, 2]
#        output_acc = dice(self.placeholders['labels'], tf.cast(self.outputs['segmentation'], tf.float32))
#
#        # mean loss across batches
#        ## output --> [2]
#        acc1 = tf.reduce_mean(output_acc, axis = 0)
#
#        # mean loss across fore/back
#        ## output --> [1]
#        acc2 = tf.reduce_mean(acc1, axis = 0)
#
#        self.outputs['dice'] = acc2

        # compute dice score (NOT loss)
        self.outputs['dice'] = dice(self.placeholders['labels'], tf.cast(self.outputs['segmentation'], tf.float32))

    ## =================================================================================================
    ## =================================================================================================

    def create_losses(self):

        if self.active_loss == 'reweighted_cross_entropy':
            self.losses['reweighted_cross_entropy'] = binary_cross_entropy_2D(
                                                                                self.placeholders['labels'], self.outputs['logits'], reweight=True
                                                                            )


        if self.active_loss == 'dice_loss':

#            ## input labels, prediction --> [None, height, width, num_classes]
#            ## output --> [None, 2]
#            output_loss = dice_loss(self.placeholders['labels'], tf.cast(self.outputs['sigmoid'], tf.float32))
#
#            # mean loss across batches
#            ## output --> [2]
#            loss1 = tf.reduce_mean(output_loss, axis = 0)
#
#            # mean loss across fore/back
#            ## output --> [1]
#            loss2 = tf.reduce_mean(loss1, axis = 0)
#
#            self.losses['dice_loss'] = loss2
#
##            print(np.asarray(output_loss).shape)
#
##            fore_dice_loss = output_loss[:,:,0]
##
##            self.losses['dice_loss'] = tf.reduce_mean(
##                                                        fore_dice_loss
##                                                    )
#
##            self.losses['dice_loss'] = tf.reduce_mean(
##                                                        dice_loss(self.placeholders['labels'], tf.cast(self.outputs['sigmoid'], tf.float32))
##                                                    )

            # dice loss
            self.losses['dice_loss'] = dice_loss(self.placeholders['labels'], tf.cast(self.outputs['sigmoid'], tf.float32))



    ## =================================================================================================
    ## =================================================================================================

    def create_training_op(self):

        # get Adam optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # put optimization routing in scope
        with tf.variable_scope("fitting_operation", reuse=tf.AUTO_REUSE):

            with tf.control_dependencies(extra_update_ops):
                self.fitting_op = self.optimizer.minimize(self.losses[self.active_loss])

    ## =================================================================================================
    ## =================================================================================================

    @timeit
    def run_epoch_train(self, curr_epoch):

#        print('\n')
        print('Training over all batches')

        self.iteration += 1

        summary_data = {
                            'sigmoid': [],
                            'loss': [],
                            'images': [],
                            'labels': [],
                            'dice': [],
                        }

        # get computed # of batches
        num_total_batches = self.num_total_batches_train
#
#        print('num_total_batches', num_total_batches)
#        print('\n')
#
#        if (self.num_total_train_images % self.batch_size) != 0:
#            num_total_batches += 1

        # prev train loss error
        dice_loss_mean = 0

        curr_batch_counter = 0

        # iterate over each batch in training set
        for batch in self.batch_iterator_train:

            # dropout is included
            if self.dropout_training_Flag == 1:

                _, loss, dice = self.run_iteration(
                                                                feed_dict={
                                                                                self.placeholders['images']: batch['images'],
                                                                                self.placeholders['labels']: batch['labels'],
                                                                                self.placeholders['is_training']: True,
                                                                                self.placeholders['dropout_prob']: self.dropout_prob_training,
                                                                            },
                                                                op_list=[
                                                                            self.fitting_op,
                                                                            self.losses[self.active_loss],
                                                                            #self.outputs['sigmoid'],
                                                                            self.outputs['dice'],
                                                                        ],
                                                                summaries=[],
                                                    )
            # no dropout
            else:

                _, loss, dice = self.run_iteration(
                                                                feed_dict={
                                                                                self.placeholders['images']: batch['images'],
                                                                                self.placeholders['labels']: batch['labels'],
                                                                                self.placeholders['is_training']: True,
                                                                            },
                                                                op_list=[
                                                                            self.fitting_op,
                                                                            self.losses[self.active_loss],
                                                                            #self.outputs['sigmoid'],
                                                                            self.outputs['dice'],
                                                                        ],
                                                                summaries=[],
                                                            )

            curr_batch_counter = curr_batch_counter + 1

            print('epoch {}, batch {} / {},  loss {},  dice {},  best_val_loss {}'.format(curr_epoch, curr_batch_counter, np.int(num_total_batches), loss, dice, self.overall_best_score))
            #print('epoch {}, loss {},  dice {}'.format(curr_epoch, loss, dice))

            summary_data['loss'].append(loss)
            summary_data['dice'].append(dice)


        dice_loss_mean = np.mean(summary_data['loss'])
        dice_score_mean = np.mean(summary_data['dice'])


        print('dice_loss_mean {},  dice_score_mean {}'.format(np.around(dice_loss_mean, self.decimal_truncate_limit), np.around(dice_score_mean, self.decimal_truncate_limit) ))

        # store summary information for the current epoch
        self.per_epoch_summary_data['epoch'].append(curr_epoch)
        self.per_epoch_summary_data['train_loss'].append(np.around(dice_loss_mean, self.decimal_truncate_limit))
        self.per_epoch_summary_data['train_dice'].append(np.around(dice_score_mean, self.decimal_truncate_limit))

#        self.saver.save(curr_epoch + 1)
#
#        if (self.iteration % 5) == 0:
#
#            print('Saving model in training session')
#
#            self.saver.save(curr_epoch + 1)

    ## =================================================================================================
    ## =================================================================================================

    @timeit
    def run_epoch_valid(self, curr_epoch):

        print('\n')
        print('Validating over all batches in current epoch')

        summary_data = {
                            'sigmoid': [],
                            'loss': [],
                            'images': [],
                            'labels': [],
                            'dice': [],
                        }

        # get computed # of batches
        num_total_batches = self.num_total_batches_valid
#        if (self.num_total_valid_images % self.batch_size) != 0:
#            num_total_batches += 1

        curr_batch_counter = 0

        # iterate over each batch in validation set
        for batch in self.batch_iterator_valid:

            # dropout is included
            if self.dropout_training_Flag == 1:

                loss, sigmoid, dice = self.run_iteration(
                                                            feed_dict={
                                                                            self.placeholders['images']: batch['images'],
                                                                            self.placeholders['labels']: batch['labels'],
                                                                            self.placeholders['is_training']: True,
                                                                            self.placeholders['dropout_prob']: self.dropout_prob_training,
                                                                        },
                                                            op_list=[
                                                                        self.losses[self.active_loss],
                                                                        self.outputs['sigmoid'],
                                                                        self.outputs['dice'],
                                                                    ],
                                                            summaries=[]
                                                    )

            else:

                loss, sigmoid, dice = self.run_iteration(
                                                            feed_dict={
                                                                            self.placeholders['images']: batch['images'],
                                                                            self.placeholders['labels']: batch['labels'],
                                                                            self.placeholders['is_training']: True,
                                                                        },
                                                            op_list=[
                                                                        self.losses[self.active_loss],
                                                                        self.outputs['sigmoid'],
                                                                        self.outputs['dice'],
                                                                    ],
                                                            summaries=[]
                                                    )

            curr_batch_counter = curr_batch_counter + 1

            print('epoch {}, batch {} / {},  loss {},  dice {}, best_val_loss {}'.format(curr_epoch, curr_batch_counter, np.int(num_total_batches), loss, dice, self.overall_best_score))


            summary_data['loss'].append(loss)
            summary_data['dice'].append(dice)


        dice_loss_mean = np.mean(summary_data['loss'])
        dice_score_mean = np.mean(summary_data['dice'])

        print('dice_loss_mean {},  dice_score_mean {}'.format(np.around(dice_loss_mean, self.decimal_truncate_limit), np.around(dice_score_mean, self.decimal_truncate_limit) ))

        # ===

#        dice_loss_mean = 0
#        dice_score_mean = 0
#
#        # smoothing alpha
#        smooth_alpha = 0.5
#
#        for loss_idx in range(len(summary_data['loss'])):
#
#            if loss_idx == 0:
#                dice_loss_mean = summary_data['loss'][loss_idx]
#                dice_score_mean = summary_data['dice'][loss_idx]
#            else:
#                dice_loss_mean = (smooth_alpha * summary_data['loss'][loss_idx]) + ((1-smooth_alpha) * dice_loss_mean)
#                dice_score_mean = (smooth_alpha * summary_data['dice'][loss_idx]) + ((1-smooth_alpha) * dice_score_mean)
#
#        print('dice_loss_mean {},  dice_score_mean {}'.format(np.around(dice_loss_mean, self.decimal_truncate_limit), np.around(dice_score_mean, self.decimal_truncate_limit) ))

        # ===

        # store summary information for the current epoch
        # already stored epoch number, just store dice loss and dice score
        self.per_epoch_summary_data['val_loss'].append(np.around(dice_loss_mean, self.decimal_truncate_limit))
        self.per_epoch_summary_data['val_dice'].append(np.around(dice_score_mean, self.decimal_truncate_limit))

        #self.add_validation_summaries(summary_data)

        if dice_loss_mean < self.overall_best_score:

            print('Saving model in validation session')

            self.saver.save(str(curr_epoch + 1) + '_best')

            self.overall_best_score = dice_loss_mean

            # reset counter for early stopping and count again
            self.counter_no_dec_in_val_loss = 0

        else:

            print('best validation score', self.overall_best_score)
            print('current validation score', dice_loss_mean)

            # increment counter for early stopping
            self.counter_no_dec_in_val_loss = self.counter_no_dec_in_val_loss + 1


        # plot the training summary (train_loss, val_loss) for every epoch
        # overwrite previous figure
        self.__plot_training_summary_graph()

        return self.counter_no_dec_in_val_loss


    ## =================================================================================================
    ## =================================================================================================


    def __plot_training_summary_graph(self):

        # get train_loss
        train_loss = self.per_epoch_summary_data['train_loss']

        # get val_loss
        val_loss = self.per_epoch_summary_data['val_loss']

        epochs = np.arange(len(train_loss)) + 1

        # plot
        fig = plt.figure()
        plt.plot(epochs, train_loss, 'rx-', label='train_loss')
        plt.plot(epochs, val_loss, 'bo--', label='val_loss')
        fig.tight_layout()

        ffp_figToSave = self.summary_path + '/' + 'trainingSummary.png'
        plt.savefig(ffp_figToSave , bbox_inches='tight')
        plt.close('all')

    ## =================================================================================================
    ## =================================================================================================

    def save_training_summaries(self):

        fn_training_summary = 'training_summary.xlsx'

        excel_ffn = os.path.join(self.summary_path, fn_training_summary)

        writer = pd.ExcelWriter(excel_ffn, engine='xlsxwriter')

        num_epochs = len(self.per_epoch_summary_data['epoch'])

        df1 = {}
        df1['Epoch'] = np.arange(1, num_epochs + 1)
        df1['train_loss'] = self.per_epoch_summary_data['train_loss']
        df1['train_dice'] = self.per_epoch_summary_data['train_dice']
        df1['val_loss'] = self.per_epoch_summary_data['val_loss']
        df1['val_dice'] = self.per_epoch_summary_data['val_dice']

        # convert to pandas dataframe
        df1 = pd.DataFrame(df1)
        # write the dataframe to excel
        df1.to_excel(writer, sheet_name = 'train_summary')

        print('path', os.path.join(self.summary_path, fn_training_summary))
        print('saving summary to disk')

        # write
        writer.save()
