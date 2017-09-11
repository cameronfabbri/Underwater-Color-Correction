'''

   Testing the features learned by the discriminator.

   This will take the features from the discriminator and attempt to classify
   images.

'''

import cPickle as pickle
import tensorflow as tf
from scipy import misc
from tqdm import tqdm
import numpy as np
import argparse
import random
import ntpath
import sys
import os
import time
import glob
import cPickle as pickle
from tqdm import tqdm

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *
import data_ops

def netD_feature(x, LAYER_NORM, LOSS_METHOD, reuse=False):
   print
   print 'netD'

   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      if LOSS_METHOD != 'wgan': conv1 = tcl.batch_norm(conv1)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv1 = tcl.layer_norm(conv1)
      conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      if LOSS_METHOD != 'wgan': conv2 = tcl.batch_norm(conv2)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv2 = tcl.layer_norm(conv2)
      conv2 = lrelu(conv2)
      
      conv3 = tcl.conv2d(conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      if LOSS_METHOD != 'wgan': conv3 = tcl.batch_norm(conv3)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv3 = tcl.layer_norm(conv3)
      conv3 = lrelu(conv3)
      
      conv4 = tcl.conv2d(conv3, 512, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      if LOSS_METHOD != 'wgan': conv4 = tcl.batch_norm(conv4)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv4 = tcl.layer_norm(conv4)
      conv4 = lrelu(conv4)
      
      conv5 = tcl.conv2d(conv4, 1, 1, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')
      if LOSS_METHOD != 'wgan': conv5 = tcl.batch_norm(conv5)
      elif LOSS_METHOD == 'wgan' and LAYER_NORM: conv5 = tcl.layer_norm(conv5)

      conv1 = tcl.max_pool2d(conv1, [4,4], stride=2, padding='SAME')
      conv1 = tcl.max_pool2d(conv1, [4,4], stride=2, padding='SAME')
      conv1 = tcl.max_pool2d(conv1, [4,4], stride=2, padding='SAME')
      conv1 = tcl.max_pool2d(conv1, [4,4], stride=2, padding='SAME')
      conv1 = tcl.max_pool2d(conv1, [4,4], stride=2, padding='SAME')
      conv2 = tcl.max_pool2d(conv2, [4,4], stride=2, padding='SAME')
      conv2 = tcl.max_pool2d(conv2, [4,4], stride=2, padding='SAME')
      conv2 = tcl.max_pool2d(conv2, [4,4], stride=2, padding='SAME')
      conv2 = tcl.max_pool2d(conv2, [4,4], stride=2, padding='SAME')
      conv3 = tcl.max_pool2d(conv3, [4,4], stride=2, padding='SAME')
      conv3 = tcl.max_pool2d(conv3, [4,4], stride=2, padding='SAME')
      conv3 = tcl.max_pool2d(conv3, [4,4], stride=2, padding='SAME')
      conv4 = tcl.max_pool2d(conv4, [4,4], stride=2, padding='SAME')
      conv4 = tcl.max_pool2d(conv4, [4,4], stride=2, padding='SAME')
      conv4 = tcl.max_pool2d(conv4, [4,4], stride=2, padding='SAME')
      conv5 = tcl.max_pool2d(conv5, [4,4], stride=2, padding='SAME')
      conv5 = tcl.max_pool2d(conv5, [4,4], stride=2, padding='SAME')
      conv5 = tcl.max_pool2d(conv5, [4,4], stride=2, padding='SAME')

      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      print
      
      # flatten all and concatenate
      conv1 = tcl.flatten(conv1)
      conv2 = tcl.flatten(conv2)
      conv3 = tcl.flatten(conv3)
      conv4 = tcl.flatten(conv4)
      conv5 = tcl.flatten(conv5)
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5

      feature_vector = tf.squeeze(tf.concat([conv1, conv2, conv3, conv4, conv5], axis=1))
      print 'feature_vector:',feature_vector

      return feature_vector

if __name__ == '__main__':

   if len(sys.argv) < 2:
      print 'You must provide an info.pkl file'
      exit()

   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)

   LEARNING_RATE = a['LEARNING_RATE']
   LOSS_METHOD   = a['LOSS_METHOD']
   BATCH_SIZE    = a['BATCH_SIZE']
   EPOCHS        = a['EPOCHS']
   L1_WEIGHT     = a['L1_WEIGHT']
   IG_WEIGHT     = a['IG_WEIGHT']
   NETWORK       = a['NETWORK']
   DATA          = a['DATA']
   LAYER_NORM    = a['LAYER_NORM']

   EXPERIMENT_DIR = 'checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                     +'/NETWORK_'+NETWORK\
                     +'/LAYER_NORM_'+str(LAYER_NORM)\
                     +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                     +'/IG_WEIGHT_'+str(IG_WEIGHT)\
                     +'/DATA_'+DATA+'/'\

   print
   print 'LEARNING_RATE: ',LEARNING_RATE
   print 'LOSS_METHOD:   ',LOSS_METHOD
   print 'BATCH_SIZE:    ',BATCH_SIZE
   print 'NETWORK:       ',NETWORK
   print 'EPOCHS:        ',EPOCHS
   print 'LAYER_NORM:    ',LAYER_NORM
   print

   if NETWORK == 'pix2pix': from pix2pix import *
   if NETWORK == 'resnet':  from resnet import *

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # underwater image
   image_u = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name='image_u')


   # generated corrected colors
   gen_image = netG(image_u, LOSS_METHOD)
   
   feature_vector = netD_feature(gen_image, LAYER_NORM, LOSS_METHOD)

   saver = tf.train.Saver(max_to_keep=1)

   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   step = int(sess.run(global_step))

   # testing paths
   test_paths = np.asarray(glob.glob('datasets/'+DATA+'/test/*.jpg'))

   print test_paths[0]
   exit()

   num_test = len(test_paths)

   print 'num test:',num_test

   c = 0
   for img_path in tqdm(test_paths):

      img_name = ntpath.basename(img_path)

      img_name = img_name.split('.')[0]

      batch_images = np.empty((1, 256, 256, 3), dtype=np.float32)

      a_img = misc.imread(img_path).astype('float32')
      a_img = misc.imresize(a_img, (256, 256, 3))
      a_img = data_ops.preprocess(a_img)
      batch_images[0, ...] = a_img

      gen_images = np.asarray(sess.run(gen_image, feed_dict={image_u:batch_images}))

