'''

   Main training file

   The goal is to correct the colors in underwater images.
   CycleGAN was used to create images that appear to be underwater.
   Those will be sent into the generator, which will attempt to correct the
   colors.

'''

import cPickle as pickle
import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import ntpath
import pix2pix
import sys
import os
import time
import glob
import cPickle as pickle

sys.path.insert(0, 'ops/')
from tf_ops import *

import data_ops

if __name__ == '__main__':

   if len(sys.argv) < 2:
      print 'You must provide an info.pkl file'
      exit()

   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)

   LEARNING_RATE = a['LEARNING_RATE']
   LOSS_METHOD   = a['LOSS_METHOD']
   L1_WEIGHT     = a['L1_WEIGHT']
   NETWORK       = a['NETWORK']
   DATA          = a['DATA']

   EXPERIMENT_DIR = 'checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                     +'/NETWORK_'+NETWORK\
                     +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                     +'/DATA_'+DATA+'/'\

   IMAGES_DIR     = EXPERIMENT_DIR+'test_images/'

   print
   print 'Creating',IMAGES_DIR
   try: os.makedirs(IMAGES_DIR)
   except: pass
   
   print
   print 'LEARNING_RATE: ',LEARNING_RATE
   print 'LOSS_METHOD:   ',LOSS_METHOD
   print 'BATCH_SIZE:    ',BATCH_SIZE
   print 'NETWORK:       ',NETWORK
   print 'EPOCHS:        ',EPOCHS
   print

   if NETWORK == 'pix2pix': from pix2pix import *
   if NETWORK == 'resnet':  from resnet import *

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # underwater image
   image_u = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='image_u')

   # correct image
   image_r = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='image_r')

   # generated corrected colors
   gen_image = netG(image_u)

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

   # underwater photos
   trainA_paths = np.asarray(glob.glob('datasets/'+DATA+'/trainA/*.jpg'))

   # normal photos (ground truth)
   trainB_paths = np.asarray(glob.glob('datasets/'+DATA+'/trainB/*.jpg'))
   
   # testing paths
   test_paths = np.asarray(glob.glob('datasets/'+DATA+'/testA/*.jpg'))

   print len(trainB_paths)
   print len(trainA_paths)

   num_train = len(trainB_paths)


   while True:

      idx = np.random.choice(np.arange(num_train), BATCH_SIZE, replace=False)
      batchA_paths = trainA_paths[idx]
      batchB_paths = trainB_paths[idx]
      
      batchA_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
      batchB_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)

      i = 0
      for a,b in zip(batchA_paths, batchB_paths):
         a_img = data_ops.preprocess(misc.imread(a).astype('float32'))
         b_img = data_ops.preprocess(misc.imread(b).astype('float32'))
         batchA_images[i, ...] = a_img
         batchB_images[i, ...] = b_img
         i += 1

      sess.run(D_train_op, feed_dict={image_u:batchA_images, image_r:batchB_images})
      sess.run(G_train_op, feed_dict={image_u:batchA_images, image_r:batchB_images})
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={image_u:batchA_images, image_r:batchB_images})

      summary_writer.add_summary(summary, step)
      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss
      step += 1
      
      
      
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'

         idx = np.random.choice(np.arange(num_train), BATCH_SIZE, replace=False)
         batchA_paths = trainA_paths[idx]
         batchB_paths = trainB_paths[idx]
         
         batchA_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)
         batchB_images = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float32)

         i = 0
         for a,b in zip(batchA_paths, batchB_paths):
            a_img = data_ops.preprocess(misc.imread(a).astype('float32'))
            b_img = data_ops.preprocess(misc.imread(b).astype('float32'))
            batchA_images[i, ...] = a_img
            batchB_images[i, ...] = b_img
            i += 1

         gen_images = np.asarray(sess.run(gen_image, feed_dict={image_u:batchA_images, image_r:batchB_images}))


         for gen, real, cor in zip(gen_images, batchB_images, batchA_images):
            misc.imsave(IMAGES_DIR+str(step)+'_corrupt.png', cor)
            misc.imsave(IMAGES_DIR+str(step)+'_real.png', real)
            misc.imsave(IMAGES_DIR+str(step)+'_gen.png', gen)
            break

