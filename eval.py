import tensorflow as tf
import numpy as np
import cPickle as pickle
import random
import ntpath
import sys
import cv2
import os
import time

from scipy import misc
from skimage import color

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'config/')

import data_ops

if __name__ == '__main__':
   
   if len(sys.argv) < 2:
      print 'You must provide an info.pkl file'
      exit()

   global_step = tf.Variable(0, name='global_step', trainable=False)

   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)

   PRETRAIN_EPOCHS = a['PRETRAIN_EPOCHS']
   GAN_EPOCHS      = a['GAN_EPOCHS']
   ARCHITECTURE    = a['ARCHITECTURE']
   DATASET         = a['DATASET']
   DATA_DIR        = a['DATA_DIR']
   PRETRAIN_LR     = a['PRETRAIN_LR']
   GAN_LR          = a['GAN_LR']
   NUM_GPU         = a['NUM_GPU']
   LOSS_METHOD     = a['LOSS_METHOD']
   NUM_CRITIC      = a['NUM_CRITIC']
   BATCH_SIZE      = a['BATCH_SIZE']
   JITTER          = a['JITTER']
   SIZE            = a['SIZE']
   L1_WEIGHT       = a['L1_WEIGHT']
   L2_WEIGHT       = a['L2_WEIGHT']
   GAN_WEIGHT      = a['GAN_WEIGHT']
   UPCONVS         = a['UPCONVS'] 

   EXPERIMENT_DIR = 'checkpoints/'+ARCHITECTURE+'_'+DATASET+'_'+LOSS_METHOD+'_'+str(PRETRAIN_EPOCHS)+'_'+str(GAN_EPOCHS)+'_'+str(PRETRAIN_LR)+'_'+str(NUM_CRITIC)+'_'+str(GAN_LR)+'_'+str(JITTER)+'_'+str(SIZE)+'_'+str(L1_WEIGHT)+'_'+str(L2_WEIGHT)+'_'+str(GAN_WEIGHT)+'_'+str(UPCONVS)+'/'
   IMAGES_DIR = EXPERIMENT_DIR+'images/'
   
   #DATASET = 'true_gray'
   print
   print 'PRETRAIN_EPOCHS: ',PRETRAIN_EPOCHS
   print 'GAN_EPOCHS:      ',GAN_EPOCHS
   print 'ARCHITECTURE:    ',ARCHITECTURE
   print 'LOSS_METHOD:     ',LOSS_METHOD
   print 'PRETRAIN_LR:     ',PRETRAIN_LR
   print 'DATASET:         ',DATASET
   print 'GAN_LR:          ',GAN_LR
   print 'NUM_GPU:         ',NUM_GPU
   print 'L1_WEIGHT:       ',L1_WEIGHT
   print 'L2_WEIGHT:       ',L2_WEIGHT
   print 'GAN_WEIGHT:      ',GAN_WEIGHT
   print 'UPCONVS:         ',UPCONVS
   print

   Data = data_ops.loadData(DATA_DIR, DATASET, BATCH_SIZE, train=False, SIZE=SIZE)
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   test_L = Data.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = Data.targets

   if ARCHITECTURE == 'pix2pix':
      import pix2pix
      predict_ab = pix2pix.netG(test_L, 0, UPCONVS)
   if ARCHITECTURE == 'colcolgan':
      import ColColGAN
      z = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 1))
      predict_ab = ColColGAN.netG(test_L, 0)

   # reconstruct prediction image from test_L and predict_ab
   prediction = data_ops.augment(predict_ab, test_L)
   prediction = tf.image.convert_image_dtype(prediction, dtype=tf.uint8, saturate=True)
   
   # reconstruct original image from test_L and ab_image
   true_image = data_ops.augment(ab_image, test_L)
   true_image = tf.image.convert_image_dtype(true_image, dtype=tf.uint8, saturate=True)
   
   saver = tf.train.Saver()
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session()
   sess.run(init)
   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         raise
         print "Could not restore model"
         exit()
   
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   import time
   colored = sess.run(prediction)
   start = time.time()
   true_   = sess.run(true_image)
   end = time.time()
   print end-start
   #exit()
   step = sess.run(global_step)

   # save out both
   i = 0
   for c,p in zip(colored, true_):
      misc.imsave(IMAGES_DIR+str(step)+'_'+str(i)+'_col.png', c)
      misc.imsave(IMAGES_DIR+str(step)+'_'+str(i)+'_true.png', p)
      if i == 4: break
      i += 1
   exit()
