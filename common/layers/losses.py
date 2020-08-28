# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:40:39 2019

@author: Tejas
"""

import tensorflow as tf

EPS = 0.0000001


def dice(labels, prediction):

    with tf.variable_scope('dice'):

        ## input --> [batch_size (None), height, width, num_classes]
        ## output --> [batch_size, num_classes]
        dc = 2.0 * \
             tf.reduce_sum(labels * prediction, axis=[1, 2]) / \
             (tf.reduce_sum(labels ** 2 + prediction ** 2, axis=[1, 2]) + EPS)

        

        ## input --> [batch_size, num_classes]
        ## output --> [num_classes]
        loss1 = tf.reduce_mean(dc, axis = 0)

        ## input --> [num_classes]
        ## output --> [1]
        loss2 = tf.reduce_mean(loss1, axis = 0)

        dc = loss2

        ## input labels, prediction --> [None, height, width, num_classes]
#        print('labels', labels.get_shape().as_list())
#        print('prediction', prediction.get_shape().as_list())

#        ## input labels, prediction --> [None, height, width, num_classes]
#        ## output --> [None, 2]
#        num = tf.reduce_sum(labels * prediction, axis=[1, 2])
#
##        print('num', num.get_shape().as_list())
#
#        ## output --> [None, 2]
##        denom = tf.reduce_sum(labels ** 2 + prediction ** 2, axis=[1, 2])
#        denom = tf.reduce_sum(labels, axis=[1, 2]) + tf.reduce_sum(prediction, axis=[1, 2])
#
##        print('denom', denom.get_shape().as_list())
#
#        ## output --> [None, 2]
#        dc = ((2.0 * num) / (denom + EPS))
#
##        print('dc', dc.get_shape().as_list())

    return dc


def dice_loss(labels, prediction):

    with tf.variable_scope('dice_loss'):

        ## input labels, prediction --> [None, height, width, num_classes]

#        print('labels', labels.get_shape().as_list())
#        print('prediction', prediction.get_shape().as_list())

        ## input labels, prediction --> [None, height, width, num_classes]
        ## output --> [1]
        diceScore = dice(labels, prediction)

#        print('diceScore', diceScore.get_shape().as_list())

        ## output --> [1]
        dl = 1.0 - diceScore

#        print('dl', dl.get_shape().as_list())

    return dl


def binary_cross_entropy_2D(labels, logits, reweight=False):
    labels_shape = labels.get_shape().as_list()

    pixel_size = labels_shape[1] * labels_shape[2]

    logits = tf.reshape(logits, [-1, pixel_size])

    labels = tf.reshape(labels, [-1, pixel_size])

    number_foreground = tf.reduce_sum(labels)
    number_background = tf.reduce_sum(1.0 - labels)

    weight_foreground = number_background / (number_foreground + EPS)
    if reweight:
        loss = \
            tf.nn.weighted_cross_entropy_with_logits(
                targets=tf.cast(labels, tf.float32),
                logits=logits,
                pos_weight=weight_foreground
            )

    else:
        loss = \
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.float32),
                logits=logits,
            )

    loss = tf.reduce_mean(loss)

    return loss



def mse(labels, prediction):

    with tf.variable_scope('dice'):

        err = labels - prediction

        sq_err = tf.square(err)

        ## output --> [None, 2]
        mse_im = tf.reduce_mean(sq_err, axis=[1, 2])

        ## output --> [2]
        loss1 = tf.reduce_mean(mse_im, axis = 0)

        loss2 = tf.reduce_mean(loss1, axis = 0)

        msev = loss2

    return msev


def mse_loss(labels, prediction):

    with tf.variable_scope('dice_loss'):

        mseScore = mse(labels, prediction)
        msel = mseScore

    return msel
