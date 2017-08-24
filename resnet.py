import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

def resBlock(x, num):
   
   conv1 = tcl.conv2d(x, 64, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_resconv1_'+str(num))
   print 'res_conv1:',conv1

   conv2 = tcl.conv2d(conv1, 64, 3, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_resconv2_'+str(num))
   print 'res_conv2:',conv2
   
   output = tf.add(x,conv2)
   print 'res_out:',output
   return output


def netG(x):
      

   conv1 = tcl.conv2d(x, 64, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')

   res1 = resBlock(conv1, 1)
   res2 = resBlock(res1, 2)
   res3 = resBlock(res2, 3)
   res4 = resBlock(res3, 4)
   res5 = resBlock(res4, 5)
   res6 = resBlock(res5, 6)
   res7 = resBlock(res6, 7)
   res8 = resBlock(res7, 8)
   res9 = resBlock(res8, 9)
   res10 = resBlock(res9, 10)
   res11 = resBlock(res10, 11)
   res12 = resBlock(res10, 12)
   res13 = resBlock(res10, 13)
   res14 = resBlock(res10, 14)
   res15 = resBlock(res10, 15)
   res16 = resBlock(res10, 16)

   conv2 = tcl.conv2d(res16, 64, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
   conv2 = tf.add(conv1, conv2)
   print 'conv2:',conv2

   conv3 = tcl.conv2d(conv2, 256, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')
   print 'conv3:',conv3
   
   conv4 = tcl.conv2d(conv3, 256, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv4')
   print 'conv4:',conv4

   conv5 = tcl.conv2d(conv4, 256, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv5')
   print 'conv5:',conv5

   conv6 = tcl.conv2d(conv5, 3, 9, 1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv6')
   print 'conv6:',conv6

   return conv6


def netD(x, reuse=False):
   print
   print 'netD'
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      conv1 = lrelu(conv1)

      conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      conv2 = lrelu(conv1)
      
      conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      conv3 = lrelu(conv3)
      
      conv4 = tcl.conv2d(conv3, 512, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      conv4 = lrelu(conv4)
      
      conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')
      
      print conv1
      print conv2
      print conv3
      print conv4
      print conv5
      return conv5


