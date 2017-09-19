import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

def lrelu(x, leak=0.2):
   return tf.maximum(leak*x, x)

def network(x, num_classes=2, is_training=True):
      
   conv = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv1')
   conv = lrelu(conv)
   print conv
   conv = tcl.conv2d(conv, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv2')
   conv = lrelu(conv)
   print conv
   conv = tcl.conv2d(conv, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')
   conv = lrelu(conv)
   print conv
   conv = tcl.conv2d(conv, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv4')
   conv = lrelu(conv)
   print conv
   conv = tcl.conv2d(conv, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv5')
   conv = lrelu(conv)
   print conv
   conv = tcl.conv2d(conv, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv6')
   conv = lrelu(conv)
   print conv
   conv = tcl.flatten(conv)
   conv = lrelu(conv)
   print conv

   fc = tcl.fully_connected(conv, 4096, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc1')
   fc = lrelu(fc)
   fc = tcl.dropout(fc, keep_prob=0.5)
   print fc
   
   fc = tcl.fully_connected(fc, 4096, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc2')
   fc = lrelu(fc)
   fc = tcl.dropout(fc, keep_prob=0.5)
   print fc

   fc = tcl.fully_connected(fc, 2048, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc3')
   fc = lrelu(fc)
   print fc

   fc = tcl.fully_connected(fc, num_classes, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc4')
   fc = tf.nn.softmax(fc)
   print fc

   return fc
