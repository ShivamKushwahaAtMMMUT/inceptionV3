# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 01:12:14 2018

@author: Shivam Kushwaha
"""

import tensorflow as tf

nodes = dict()

def conv2d(x, filter_size, strides, num_filters, padding, scope):
    num_channels = x.get_shape().as_list()[-1]
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
    b_init = tf.constant_initializer(0.0)
    w = tf.get_variable("weight_"+scope, 
                        [filter_size[0], filter_size[1], 
                         num_channels, num_filters], 
                         tf.float32, w_init)
    b = tf.get_variable("bias_"+scope, [num_filters], tf.float32, b_init)
    conv = tf.nn.conv2d(x, w, [1, strides, strides, 1], padding)
    return tf.nn.relu(tf.nn.bias_add(conv, b))


def max_pool(x, ksize, strides, padding, scope):
    return tf.nn.max_pool(x, [1, ksize[0], ksize[1], 1], 
                          [1, strides, strides, 1], 
                          padding, name="max_pool_"+scope)


def avg_pool(x, ksize, strides, padding, scope):
    return tf.nn.avg_pool(x, [1, ksize[0], ksize[1], 1], 
                          [1, strides, strides, 1], 
                          padding, name="avg_pool_"+scope)


def fully_connected(x, num_outputs, activation, scope):
    shape = x.get_shape().as_list()
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.1)
    b_init = tf.constant_initializer(0.0)
    w = tf.get_variable("weight_"+scope, [shape[1], num_outputs], tf.float32, 
                        w_init)
    b = tf.get_variable("bias_"+scope, [num_outputs], tf.float32, b_init)
    mul = tf.matmul(x, w) + b
    if not activation == None:
        mul = activation(mul)
    return mul


def inception_v3_base(x):
    end_points = dict()
    
    with tf.variable_scope("conv2d_1a"):
        conv1 = conv2d(x, (3, 3), 2, 32, "VALID", "1")
    end_points["conv2d_1a"] = conv1
    with tf.variable_scope("conv2d_2a"):
        conv1 = conv2d(conv1, (3, 3), 1, 32, "VALID", "1")
    end_points["conv2d_2a"] = conv1
    with tf.variable_scope("conv2d_3a"):
        conv1 = conv2d(conv1, (3, 3), 1, 64, "SAME", "1")
    end_points["conv2d_3a"] = conv1
    with tf.variable_scope("maxpool_4a"):
        mp1 = max_pool(conv1, (3, 3), 2, "VALID", "1")
    end_points["maxpool_4a"] = mp1
    with tf.variable_scope("conv2d_5a"):
        conv1 = conv2d(mp1, (3, 3), 1, 80, "VALID", "1")
    end_points["conv2d_5a"] = conv1
    with tf.variable_scope("conv2d_6a"):
        conv1 = conv2d(conv1, (3, 3), 2, 192, "VALID", "1")
    end_points["conv2d_6a"] = conv1
    with tf.variable_scope("conv2d_7a"):
        conv1 = conv2d(conv1, (3, 3), 1, 288, "SAME", "1")
    end_points["conv2d_7a"] = conv1
    
    # 3x Module A
    # in_channels = 288
    # out_channels = 768
    with tf.variable_scope("moduleA_8a"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 128, "SAME", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 168, "SAME", "1")
            branch2 = conv2d(branch2, (5, 5), 1, 224, "SAME", "2")
        with tf.variable_scope("branch_3"):
            branch3 = conv2d(conv1, (1, 1), 1, 192, "SAME", "1")
            branch3 = conv2d(branch3, (3, 3), 1, 288, "SAME", "2")
            branch3 = conv2d(branch3, (3, 3), 1, 288, "SAME", "3")
        with tf.variable_scope("branch_4"):
            branch4 = avg_pool(conv1, (3, 3), 1, "SAME", "1")
            branch4 = conv2d(branch4, (1, 1), 1, 128, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2, branch3, branch4], 
                          axis=3)
    end_points["moduleA_8a"] = conv1
    with tf.variable_scope("moduleA_8b"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 128, "SAME", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 168, "SAME", "1")
            branch2 = conv2d(branch2, (5, 5), 1, 224, "SAME", "2")
        with tf.variable_scope("branch_3"):
            branch3 = conv2d(conv1, (1, 1), 1, 192, "SAME", "1")
            branch3 = conv2d(branch3, (3, 3), 1, 288, "SAME", "2")
            branch3 = conv2d(branch3, (3, 3), 1, 288, "SAME", "3")
        with tf.variable_scope("branch_4"):
            branch4 = avg_pool(conv1, (3, 3), 1, "SAME", "1")
            branch4 = conv2d(branch4, (1, 1), 1, 128, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2, branch3, branch4], 
                          axis=3)
    end_points["moduleA_8b"] = conv1
    with tf.variable_scope("moduleA_8c"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 128, "SAME", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 168, "SAME", "1")
            branch2 = conv2d(branch2, (5, 5), 1, 224, "SAME", "2")
        with tf.variable_scope("branch_3"):
            branch3 = conv2d(conv1, (1, 1), 1, 192, "SAME", "1")
            branch3 = conv2d(branch3, (3, 3), 1, 288, "SAME", "2")
            branch3 = conv2d(branch3, (3, 3), 1, 288, "SAME", "3")
        with tf.variable_scope("branch_4"):
            branch4 = avg_pool(conv1, (3, 3), 1, "SAME", "1")
            branch4 = conv2d(branch4, (1, 1), 1, 128, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2, branch3, branch4], 
                          axis=3)
    end_points["moduleA_8c"] = conv1
    
    # 1x module C
    # in_channels = 768
    # out_channels = 768
    with tf.variable_scope("moduleC_9a"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (3, 3), 2, 384, "VALID", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 64, "SAME", "1")
            branch2 = conv2d(branch2, (3, 3), 1, 96, "SAME", "2")
            branch2 = conv2d(branch2, (3, 3), 2, 96, "VALID", "3")
        with tf.variable_scope("branch_3"):
            branch3 = max_pool(conv1, (3, 3), 2, "VALID", "1")
            branch3 = max_pool(branch3, (1, 1), 1, 288, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2, branch3], axis=3)
    end_points["moduleC_9a"] = conv1
    
    # 4x module B
    # in_channels = 768
    # out_channels = 1280
    with tf.variable_scope("moduleB_10a"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 288, "SAME", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 256, "SAME", "1")
            branch2 = conv2d(branch2, (1, 7), 1, 256, "SAME", "2")
            branch2 = conv2d(branch2, (7, 1), 1, 352, "SAME", "3")
        with tf.variable_scope("branch_3"):
            branch3 = conv2d(conv1, (1, 1), 1, 256, "SAME", "1")
            branch3 = conv2d(branch3, (7, 1), 1, 256, "SAME", "2")
            branch3 = conv2d(branch3, (1, 7), 1, 256, "SAME", "3")
            branch3 = conv2d(branch3, (7, 1), 1, 256, "SAME", "4")
            branch3 = conv2d(branch3, (1, 7), 1, 352, "SAME", "5")
        with tf.variable_scope("branch_4"):
            branch4 = avg_pool(conv1, (3, 3), 1, "SAME", "1")
            branch4 = conv2d(branch4, (1, 1), 1, 288, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)
    end_points["moduleB_10a"] = conv1
    with tf.variable_scope("moduleB_10b"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 288, "SAME", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 320, "SAME", "1")
            branch2 = conv2d(branch2, (1, 7), 1, 320, "SAME", "2")
            branch2 = conv2d(branch2, (7, 1), 1, 352, "SAME", "3")
        with tf.variable_scope("branch_3"):
            branch3 = conv2d(conv1, (1, 1), 1, 320, "SAME", "1")
            branch3 = conv2d(branch3, (7, 1), 1, 320, "SAME", "2")
            branch3 = conv2d(branch3, (1, 7), 1, 320, "SAME", "3")
            branch3 = conv2d(branch3, (7, 1), 1, 320, "SAME", "4")
            branch3 = conv2d(branch3, (1, 7), 1, 352, "SAME", "5")
        with tf.variable_scope("branch_4"):
            branch4 = avg_pool(conv1, (3, 3), 1, "SAME", "1")
            branch4 = conv2d(branch4, (1, 1), 1, 288, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)
    end_points["moduleB_10b"] = conv1
    with tf.variable_scope("moduleB_10c"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 288, "SAME", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 352, "SAME", "1")
            branch2 = conv2d(branch2, (1, 7), 1, 352, "SAME", "2")
            branch2 = conv2d(branch2, (7, 1), 1, 352, "SAME", "3")
        with tf.variable_scope("branch_3"):
            branch3 = conv2d(conv1, (1, 1), 1, 352, "SAME", "1")
            branch3 = conv2d(branch3, (7, 1), 1, 352, "SAME", "2")
            branch3 = conv2d(branch3, (1, 7), 1, 352, "SAME", "3")
            branch3 = conv2d(branch3, (7, 1), 1, 352, "SAME", "4")
            branch3 = conv2d(branch3, (1, 7), 1, 352, "SAME", "5")
        with tf.variable_scope("branch_4"):
            branch4 = avg_pool(conv1, (3, 3), 1, "SAME", "1")
            branch4 = conv2d(branch4, (1, 1), 1, 288, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)
    end_points["moduleB_10c"] = conv1
    with tf.variable_scope("moduleB_10d"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 288, "SAME", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 352, "SAME", "1")
            branch2 = conv2d(branch2, (1, 7), 1, 352, "SAME", "2")
            branch2 = conv2d(branch2, (7, 1), 1, 352, "SAME", "3")
        with tf.variable_scope("branch_3"):
            branch3 = conv2d(conv1, (1, 1), 1, 352, "SAME", "1")
            branch3 = conv2d(branch3, (7, 1), 1, 352, "SAME", "2")
            branch3 = conv2d(branch3, (1, 7), 1, 352, "SAME", "3")
            branch3 = conv2d(branch3, (7, 1), 1, 352, "SAME", "4")
            branch3 = conv2d(branch3, (1, 7), 1, 352, "SAME", "5")
        with tf.variable_scope("branch_4"):
            branch4 = avg_pool(conv1, (3, 3), 1, "SAME", "1")
            branch4 = conv2d(branch4, (1, 1), 1, 288, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)
    end_points["moduleB_10d"] = conv1
    
    # 1x module D
    # in_channels = 1280
    # out_channels = 1280
    with tf.variable_scope("moduleD_11a"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 192, "SAME", "1")
            branch1 = conv2d(branch1, (3, 3), 2, 320, "VALID", "2")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 256, "SAME", "1")
            branch2 = conv2d(branch2, (1, 7), 1, 256, "SAME", "2")
            branch2 = conv2d(branch2, (7, 1), 1, 256, "SAME", "3")
            branch2 = conv2d(branch2, (3, 3), 2, 320, "VALID", "4")
        with tf.variable_scope("branch_3"):
            branch3 = max_pool(conv1, (3, 3), 2, "VALID", "1")
            branch3 = conv2d(branch3, (1, 1), 1, 640, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2, branch3], axis=3)
    end_points["moduleD_11a"] = conv1
    
    # 2x module E
    # in_channels = 1280
    # out_channels = 2048
    with tf.variable_scope("moduleE_12a"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 320, "SAME", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 384, "SAME", "1")
            branch2_a = conv2d(branch2, (3, 1), 1, 384, "SAME", "2")
            branch2_b = conv2d(branch2, (1, 3), 1, 384, "SAME", "3")
        with tf.variable_scope("branch_3"):
            branch3 = conv2d(conv1, (1, 1), 1, 448, "SAME", "1")
            branch3 = conv2d(branch3, (3, 3), 1, 384, "SAME", "2")
            branch3_a = conv2d(branch3, (3, 1), 1, 384, "SAME", "3")
            branch3_b = conv2d(branch3, (1, 3), 1, 384, "SAME", "4")
        with tf.variable_scope("branch_4"):
            branch4 = avg_pool(conv1, (3, 3), 1, "SAME", "1")
            branch4 = conv2d(branch4, (1, 1), 1, 192, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2_a, branch2_b, 
                                  branch3_a, branch3_b, branch4], axis=3)
    end_points["moduleE_12a"] = conv1
    with tf.variable_scope("moduleE_12b"):
        with tf.variable_scope("branch_1"):
            branch1 = conv2d(conv1, (1, 1), 1, 320, "SAME", "1")
        with tf.variable_scope("branch_2"):
            branch2 = conv2d(conv1, (1, 1), 1, 384, "SAME", "1")
            branch2_a = conv2d(branch2, (3, 1), 1, 384, "SAME", "2")
            branch2_b = conv2d(branch2, (1, 3), 1, 384, "SAME", "3")
        with tf.variable_scope("branch_3"):
            branch3 = conv2d(conv1, (1, 1), 1, 448, "SAME", "1")
            branch3 = conv2d(branch3, (3, 3), 1, 384, "SAME", "2")
            branch3_a = conv2d(branch3, (3, 1), 1, 384, "SAME", "3")
            branch3_b = conv2d(branch3, (1, 3), 1, 384, "SAME", "4")
        with tf.variable_scope("branch_4"):
            branch4 = avg_pool(conv1, (3, 3), 1, "SAME", "1")
            branch4 = conv2d(branch4, (1, 1), 1, 192, "SAME", "2")
        conv1 = tf.concat(values=[branch1, branch2_a, branch2_b, 
                                  branch3_a, branch3_b, branch4], axis=3)
    end_points["moduleE_12b"] = conv1
    return end_points, conv1

def inception_v3(num_classes, in_width=299, in_height=299, in_channels=3):
    input_plh = tf.placeholder(dtype=tf.float32, 
                              shape=[None, in_height, in_width, in_channels], 
                              name="input_placeholder")
    
    end_points, conv1 = inception_v3_base(input_plh)
    
    end_points["input_placeholder"] = input_plh
    
    # Auxilary Classifier
    with tf.variable_scope("auxilary_classifier"):
        aux_input = end_points["moduleB_10d"]
        aux_avg_pool = avg_pool(aux_input, (5, 5), 3, "VALID", "1")
        aux_conv = conv2d(aux_avg_pool, (1, 1), 1, 768, "SAME", "2")
        aux_s = aux_conv.get_shape().as_list()
        aux_reshaped = tf.reshape(aux_conv, [-1, aux_s[1]*aux_s[2]*aux_s[3]])
        aux_fc = fully_connected(aux_reshaped, 1024, tf.nn.relu, "3")
        aux_logits = fully_connected(aux_fc, num_classes, None, "4")
    end_points["auxilary_logits"] = aux_logits
    
    # Main Classifier
    avg_pooled = avg_pool(conv1, (8, 8), 1, "VALID", "1")
    shape = avg_pooled.get_shape().as_list()
    reshaped = tf.reshape(avg_pooled, [-1, shape[1]*shape[2]*shape[3]])
    dropped = tf.nn.dropout(reshaped, 0.8)
    fc = fully_connected(dropped, 1024, tf.nn.relu, "2")
    logits = fully_connected(fc, num_classes, None, "3")
    end_points["logits"] = logits
    
    return end_points, input_plh, logits, aux_logits

