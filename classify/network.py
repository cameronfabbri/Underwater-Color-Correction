import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

#def tf.nn.relu(x, leak=0.2):
#   return tf.maximum(leak*x, x)

def network(x, num_classes=2, is_training=True):
   print x
   conv = tcl.conv2d(x, 96, 11, 4, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv1', padding='VALID')
   conv = tf.nn.relu(conv)
   print conv

   conv = tcl.max_pool2d(conv, 3, 2)
   print conv

   conv = tcl.conv2d(conv, 256, 5, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv2')
   conv = tf.nn.relu(conv)
   print conv

   conv = tcl.max_pool2d(conv, 3, 2)
   print conv
   
   #conv = tf.pad(conv, ([0,0], [1,1], [1,1], [0,0]))
   
   conv = tcl.conv2d(conv, 384, 3, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')
   conv = tf.nn.relu(conv)
   print conv
   
   conv = tcl.conv2d(conv, 384, 3, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv4')
   conv = tf.nn.relu(conv)
   print conv
   
   conv = tcl.conv2d(conv, 256, 3, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv5')
   conv = tf.nn.relu(conv)
   print conv
   
   conv = tcl.max_pool2d(conv, 3, 2)
   print conv
   
   conv = tcl.flatten(conv)

   fc = tcl.fully_connected(conv, 4096, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc1')
   print fc

   fc = tcl.fully_connected(fc, 4096, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc2')
   print fc
   
   fc = tcl.fully_connected(fc, num_classes, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc3')
   print fc

   fc = tf.nn.softmax(fc)

   return fc

   '''
   conv = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv1')
   conv = tf.nn.relu(conv)
   print conv
   conv = tcl.max_pool2d(conv, 4)
   print conv
   conv = tcl.conv2d(conv, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv2')
   conv = tf.nn.relu(conv)
   print conv
   conv = tcl.max_pool2d(conv, 4)
   print conv
   conv = tcl.conv2d(conv, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')
   conv = tf.nn.relu(conv)
   print conv
   conv = tcl.max_pool2d(conv, 4)
   print conv
   conv = tcl.conv2d(conv, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv4')
   conv = tf.nn.relu(conv)
   print conv
   conv = tcl.conv2d(conv, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv5')
   conv = tf.nn.relu(conv)
   print conv
   conv = tcl.conv2d(conv, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv6')
   conv = tf.nn.relu(conv)
   print conv
   conv = tcl.flatten(conv)
   conv = tf.nn.relu(conv)
   print conv

   fc = tcl.fully_connected(conv, 4096, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc1')
   fc = tf.nn.relu(fc)
   #fc = tcl.dropout(fc, keep_prob=0.5)
   print fc
   
   fc = tcl.fully_connected(fc, 4096, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc2')
   fc = tf.nn.relu(fc)
   #fc = tcl.dropout(fc, keep_prob=0.5)
   print fc

   fc = tcl.fully_connected(fc, num_classes, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc4')
   fc = tf.nn.softmax(fc)
   return fc
   '''
