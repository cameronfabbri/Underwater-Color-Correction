import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys

sys.path.insert(0, 'ops/')
from tf_ops import lrelu, conv2d, batch_norm, conv2d_transpose, relu, tanh

def netG(x):
      
   enc_conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='enc_conv1')
   enc_conv1 = lrelu(enc_conv1)
   
   enc_conv2 = tcl.conv2d(enc_conv1, 128, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='enc_conv2')
   enc_conv2 = lrelu(enc_conv2)
   
   enc_conv3 = tcl.conv2d(enc_conv2, 256, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='enc_conv3')
   enc_conv3 = lrelu(enc_conv3)

   enc_conv4 = tcl.conv2d(enc_conv3, 512, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='enc_conv4')
   enc_conv4 = lrelu(enc_conv4)
   
   enc_conv5 = tcl.conv2d(enc_conv4, 512, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='enc_conv5')
   enc_conv5 = lrelu(enc_conv5)

   enc_conv6 = tcl.conv2d(enc_conv5, 512, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='enc_conv6')
   enc_conv6 = lrelu(enc_conv6)
   
   enc_conv7 = tcl.conv2d(enc_conv6, 512, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='enc_conv7')
   enc_conv7 = lrelu(enc_conv7)

   enc_conv8 = tcl.conv2d(enc_conv7, 512, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='enc_conv8')
   enc_conv8 = lrelu(enc_conv8)

   print 'x:        ',x
   print 'enc_conv1:',enc_conv1
   print 'enc_conv2:',enc_conv2
   print 'enc_conv3:',enc_conv3
   print 'enc_conv4:',enc_conv4
   print 'enc_conv5:',enc_conv5
   print 'enc_conv6:',enc_conv6
   print 'enc_conv7:',enc_conv7
   print 'enc_conv8:',enc_conv8
   print

   SKIP_CONNECTIONS = 1

   dec_conv1 = tcl.convolution2d_transpose(enc_conv8, 512, 4, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='dec_conv1')
   dec_conv1 = relu(dec_conv1)
   
   if SKIP_CONNECTIONS == 1: dec_conv1 = tf.concat([dec_conv1, enc_conv7], axis=3)
   print 'dec_conv1:',dec_conv1

   exit()


   UPCONVS = 0
   with tf.variable_scope('g_dec1'):
      print '1:',enc_conv8
      if UPCONVS:
         print 'Using up-convolutions'
         dec_convt1 = tf.image.resize_nearest_neighbor(enc_conv8, [2,2])
         dec_convt1 = conv2d(dec_convt1, 512, stride=1, kernel_size=3)
      else: dec_convt1 = conv2d_transpose(enc_conv8, 512, stride=2, kernel_size=4)
      dec_convt1 = batch_norm(dec_convt1)
      dec_convt1 = relu(dec_convt1)
      dec_convt1 = tf.nn.dropout(dec_convt1, keep_prob=0.5)
      print dec_convt1
   with tf.variable_scope('g_dec2'):
      dec_convt2 = tf.concat([dec_convt1, enc_conv7], axis=3)
      print dec_convt2
      if UPCONVS:
         dec_convt2 = tf.image.resize_nearest_neighbor(dec_convt2, [4,4])
         dec_convt2 = conv2d(dec_convt2, 512, stride=1, kernel_size=3)
      else: dec_convt2 = conv2d_transpose(dec_convt2, 512, stride=2, kernel_size=4)
      dec_convt2 = batch_norm(dec_convt2)
      dec_convt2 = relu(dec_convt2)
   with tf.variable_scope('g_dec3'):
      dec_convt3 = tf.concat([enc_conv6, dec_convt2], axis=3)
      print dec_convt3
      if UPCONVS:
         dec_convt3 = tf.image.resize_nearest_neighbor(dec_convt3, [8,8])
         dec_convt3 = conv2d(dec_convt3, 512, stride=1, kernel_size=3)
      else:
         dec_convt3 = conv2d_transpose(dec_convt3, 512, stride=2, kernel_size=4)
      dec_convt3 = batch_norm(dec_convt3)
      dec_convt3 = relu(dec_convt3)
   with tf.variable_scope('g_dec4'):
      dec_convt4 = tf.concat([enc_conv5, dec_convt3], axis=3)
      print dec_convt4
      if UPCONVS:
         dec_convt4 = tf.image.resize_nearest_neighbor(dec_convt4, [16,16])
         dec_convt4 = conv2d(dec_convt4, 512, stride=1, kernel_size=3)
      else: dec_convt4 = conv2d_transpose(dec_convt4, 512, stride=2, kernel_size=4)
      dec_convt4 = batch_norm(dec_convt4)
      dec_convt4 = relu(dec_convt4)
   with tf.variable_scope('g_dec5'):
      dec_convt5 = tf.concat([enc_conv4, dec_convt4], axis=3)
      print dec_convt5
      if UPCONVS:
         dec_convt5 = tf.image.resize_nearest_neighbor(dec_convt5, [32,32])
         dec_convt5 = conv2d(dec_convt5, 256, stride=1, kernel_size=3)
      else: dec_convt5 = conv2d_transpose(dec_convt5, 256, stride=2, kernel_size=4)
      dec_convt5 = batch_norm(dec_convt5)
      dec_convt5 = relu(dec_convt5)
   with tf.variable_scope('g_dec6'):
      dec_convt6 = tf.concat([enc_conv3, dec_convt5], axis=3)
      print dec_convt6
      if UPCONVS:
         dec_convt6 = tf.image.resize_nearest_neighbor(dec_convt6, [64,64])
         dec_convt6 = conv2d(dec_convt6, 128, stride=1, kernel_size=3)
      else: dec_convt6 = conv2d_transpose(dec_convt6, 128, stride=2, kernel_size=4)
      dec_convt6 = batch_norm(dec_convt6)
      dec_convt6 = relu(dec_convt6)
   with tf.variable_scope('g_dec7'):
      dec_convt7 = tf.concat([enc_conv2, dec_convt6], axis=3)
      print dec_convt7
      if UPCONVS:
         dec_convt7 = tf.image.resize_nearest_neighbor(dec_convt7, [128,128])
         dec_convt7 = conv2d(dec_convt7, 128, stride=1, kernel_size=3)
      else: dec_convt7 = conv2d_transpose(dec_convt7, 128, stride=2, kernel_size=4)
      dec_convt7 = batch_norm(dec_convt7)
      dec_convt7 = relu(dec_convt7)

   # output layer - ab channels
   with tf.variable_scope('g_dec8'):
      if UPCONVS:
         dec_convt8 = tf.image.resize_nearest_neighbor(dec_convt7, [256,256])
         dec_convt8 = conv2d(dec_convt8, 2, stride=1, kernel_size=3)
      else: dec_convt8 = conv2d_transpose(dec_convt7, 2, stride=2, kernel_size=4)
      dec_convt8 = tanh(dec_convt8)
      
   print dec_convt8

   return dec_convt8


def netD(x, reuse=False):
   print
   print 'netD'
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv1')
      conv1 = lrelu(conv1)

      conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv2')
      conv2 = lrelu(conv1)
      
      conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')
      conv3 = lrelu(conv3)
      
      conv4 = tcl.conv2d(conv3, 512, 4, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv4')
      conv4 = lrelu(conv4)
      
      conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv5')
      
      print conv1
      print conv2
      print conv3
      print conv4
      print conv5
      return conv5


