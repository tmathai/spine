# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:44:32 2019

@author: Tejas
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import slim

from layers.layers import BasicConvLSTMCell

BASE_NUM_KERNELS = 64


def batch_norm_relu(inputs, is_training):

    net = slim.batch_norm(inputs, is_training=is_training)
    net = tf.nn.relu(net)
    return net

def dropout (input, keep_prob, is_training):
    if is_training == True:
        dropout = tf.nn.dropout(input, keep_prob)
    else:
        dropout = input
    return dropout


def conv2d_transpose(inputs, output_channels, kernel_size):

    upsamp = tf.contrib.slim.conv2d_transpose(
                                                    inputs,
                                                    num_outputs=output_channels,
                                                    kernel_size=kernel_size,
                                                    stride=2,
                                            )
    return upsamp



def conv2d_fixed_padding(inputs, filters, kernel_size, stride):

    net = slim.conv2d(inputs,
                      filters,
                      kernel_size,
                      stride=stride,
                      padding=('SAME' if stride == 1 else 'VALID'),
                      activation_fn=None
                      )
    return net



def building_block(inputs, filters, is_training, projection_shortcut, stride):

    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training)
    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, stride=stride)

    inputs = batch_norm_relu(inputs, is_training)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, stride=1)

    return inputs + shortcut



def bottleneck_block(inputs, filters, is_training, projection_shortcut, stride):

    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, stride=1)

    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, stride=stride)

    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, stride=1)

    return inputs + shortcut



def block_layer_compressing(inputs, filters, block_fn, blocks, stride, is_training, name):

    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, stride=stride)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, stride)

    layers_outputs = [inputs]

    for i in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1)

        layers_outputs.append(tf.nn.relu(inputs))

    return tf.identity(inputs, name), layers_outputs



def block_layer_expanding(inputs, forwarded_feature_list, filters, block_fn, blocks, stride, is_training, name,
                          concat_zero=True):
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    if concat_zero:
        inputs = tf.concat([inputs, forwarded_feature_list[0]], axis=-1)

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, stride=stride)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, stride)

    layers_outputs = [inputs]

    for i in range(1, blocks):
        inputs = tf.concat([inputs, forwarded_feature_list[i]], axis=-1)
        inputs = block_fn(inputs, filters, is_training, projection_shortcut, 1)

        layers_outputs.append(tf.nn.relu(inputs))

    return tf.identity(inputs, name), layers_outputs



def seg_unet_reg_v1_dropout_generator(block_fn, layers, num_classes, keep_prob, data_format='channels_last'):

    def model(inputs, is_training):

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

            base_num_kernels = 64

            # =================================
            # 64

            x = conv2d_fixed_padding(inputs=inputs, filters=base_num_kernels, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            output_b1 = x
            output_list_b1 = [x]

            # 64
            # =================================
            # 64

            x = dropout(x, keep_prob, is_training)

            x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

            # 64
            # =================================
            # 64

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 2, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 2, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            output_b2 = x
            output_list_b2 = [x]

            # 128
            # =================================
            # 128

            x = dropout(x, keep_prob, is_training)

            x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

            # 128
            # =================================
            # 128

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 4, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 4, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            output_b3 = x
            output_list_b3 = [x]

            # 256
            # =================================
            # 256

            x = dropout(x, keep_prob, is_training)

            x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

            # 256
            # =================================
            # 256

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 8, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 8, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            output_b4 = x
            output_list_b4 = [x]

            # 512
            # =================================
            # 512

            x = dropout(x, keep_prob, is_training)

            x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

            # 512
            # =================================
            # 512

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 16, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 16, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            # 1024
            # =================================
            # 1024

            up_4 = conv2d_transpose(x, kernel_size=2, output_channels=base_num_kernels * 8)

            # 512
            # =================================
            # 512

            concat_4 = tf.concat([up_4, output_b4], axis=-1)

            # 1024
            # =================================
            # 1024

            x = conv2d_fixed_padding(inputs=concat_4, filters=base_num_kernels * 8, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 8, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            # 512
            # =================================
            # 512

            up_3 = conv2d_transpose(x, kernel_size=2, output_channels=base_num_kernels * 4)

            # 256
            # =================================
            # 256

            concat_3 = tf.concat([up_3, output_b3], axis=-1)

            # 512
            # =================================
            # 512

            x = conv2d_fixed_padding(inputs=concat_3, filters=base_num_kernels * 4, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 4, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            # 256
            # =================================
            # 256

            up_2 = conv2d_transpose(x, kernel_size=2, output_channels=base_num_kernels * 2)

            # 128
            # =================================
            # 128

            concat_2 = tf.concat([up_2, output_b2], axis=-1)

            # 256
            # =================================
            # 256

            x = conv2d_fixed_padding(inputs=concat_2, filters=base_num_kernels * 2, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 2, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            # 128
            # =================================
            # 128

            up_1 = conv2d_transpose(x, kernel_size=2, output_channels=base_num_kernels)

            # 64
            # =================================


            concat_1 = tf.concat([up_1, output_b1], axis=-1)

            # 128
            # =================================
            # 128

            x = conv2d_fixed_padding(inputs=concat_1, filters=base_num_kernels * 1, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            x = conv2d_fixed_padding(inputs=x, filters=base_num_kernels * 1, kernel_size=3, stride=1)

            x = tf.nn.relu(x)   #batch_norm_relu(x, is_training)

            # 64
            # =================================
            # output

            outputs = conv2d_fixed_padding(inputs=x, filters=num_classes, kernel_size=3, stride=1)

            if num_classes == 1:

                outputs = tf.nn.sigmoid(outputs)

            else:

    #            b = outputs.get_shape().as_list()[0]
                h = outputs.get_shape().as_list()[1]
                w = outputs.get_shape().as_list()[2]

                outputs_reshaped = tf.reshape(outputs, np.asarray([-1, num_classes]))

                outputs_final = tf.nn.softmax(outputs_reshaped)

                outputs = tf.reshape(outputs_final, np.asarray([-1, h, w, num_classes]))

            return outputs

    return model



def seg_unet_reg_v1_dropout(unet_size, num_classes, keep_prob, data_format=None):

    model_params = {
                        'unet_reg_v1': {'block': building_block, 'layers': [2, 2, 2, 2]},
                    }

    if unet_size not in model_params:
        raise ValueError('Not a valid unet_size:', unet_size)

    params = model_params[unet_size]

    return seg_unet_reg_v1_dropout_generator(params['block'], params['layers'], num_classes, keep_prob, data_format)
