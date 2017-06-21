'''

Operations commonly used in tensorflow

'''

import tensorflow as tf
import numpy as np
import math

'''
'''
def batch_norm(x):
    with tf.variable_scope('batchnorm'):
      x = tf.identity(x)

      channels = x.get_shape()[3]
      offset = tf.get_variable('offset', [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
      scale = tf.get_variable('scale', [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
      mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
      variance_epsilon = 1e-5
      normalized = tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
      return normalized

def conv2d(x, out_channels, stride=2,kernel_size=4):
   with tf.variable_scope('conv2d'):
      in_channels = x.get_shape()[3]
      kernel = tf.get_variable('kernel', [kernel_size, kernel_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
      # [batch, in_height, in_width, in_channels], [kernel_width, kernel_height, in_channels, out_channels]
      #     => [batch, out_height, out_width, out_channels]
      padded_input = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
      conv = tf.nn.conv2d(padded_input, kernel, [1, stride, stride, 1], padding='VALID')
      return conv

def conv2d_transpose(x, out_channels, stride=2, kernel_size=4):
   with tf.variable_scope('conv2d_transpose'):
      batch, in_height, in_width, in_channels = [int(d) for d in x.get_shape()]
      kernel = tf.get_variable('kernel', [kernel_size, kernel_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
      conv = tf.nn.conv2d_transpose(x, kernel, [batch, in_height * 2, in_width * 2, out_channels], [1, stride, stride, 1], padding='SAME')
      return conv


######## activation functions ###########
'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2):
   return tf.maximum(leak*x, x)
         
'''
   Regular relu
'''
def relu(x, name='relu'):
   return tf.nn.relu(x, name)

'''
   Tanh
'''
def tanh(x, name='tanh'):
   return tf.nn.tanh(x, name)

'''
   Sigmoid
'''
def sig(x, name='sig'):
   return tf.nn.sigmoid(x, name)
