'''

   This interpolates between embeddings.
   Must pass in a distorted image and it's clean pair.

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
import time
import glob
import cPickle as pickle
from tqdm import tqdm

sys.path.insert(0, '../ops/')
sys.path.insert(0, '../nets/')
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
   BATCH_SIZE    = a['BATCH_SIZE']
   L1_WEIGHT     = a['L1_WEIGHT']
   IG_WEIGHT     = a['IG_WEIGHT']
   NETWORK       = a['NETWORK']
   AUGMENT       = a['AUGMENT']
   EPOCHS        = a['EPOCHS']
   DATA          = a['DATA']

   EXPERIMENT_DIR  = '../checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                     +'/NETWORK_'+NETWORK\
                     +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                     +'/IG_WEIGHT_'+str(IG_WEIGHT)\
                     +'/AUGMENT_'+str(AUGMENT)\
                     +'/DATA_'+DATA+'/'\

   IMAGES_DIR     = EXPERIMENT_DIR+'test_images/'

   print
   print 'LEARNING_RATE: ',LEARNING_RATE
   print 'LOSS_METHOD:   ',LOSS_METHOD
   print 'BATCH_SIZE:    ',BATCH_SIZE
   print 'L1_WEIGHT:     ',L1_WEIGHT
   print 'IG_WEIGHT:     ',IG_WEIGHT
   print 'NETWORK:       ',NETWORK
   print 'EPOCHS:        ',EPOCHS
   print 'DATA:          ',DATA
   print

   if NETWORK == 'pix2pix': from pix2pix import *
   if NETWORK == 'resnet':  from resnet import *

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # underwater image
   image_u = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name='image_u')

   # generated corrected colors
   layers    = netG_encoder(image_u)
   gen_image = netG_decoder(layers)

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

   clean_images = glob.glob('clean/*.*')
   distorted_images = glob.glob('distorted/*.*')

   for cimg, dimg in zip(clean_images, distorted_images):

      batch_cimages = np.empty((1, 256, 256, 3), dtype=np.float32)
      batch_dimages = np.empty((1, 256, 256, 3), dtype=np.float32)

      img = misc.imread(cimg).astype('float32')
      img = misc.imresize(img, (256, 256, 3))
      img = data_ops.preprocess(img)
      batch_cimages[0, ...] = img
      
      img = misc.imread(dimg).astype('float32')
      img = misc.imresize(img, (256, 256, 3))
      img = data_ops.preprocess(img)
      batch_dimages[0, ...] = img

      # send through decoder and get all layers
      all_layers = sess.run(layers, feed_dict={image_u:batch_cimages})
      print all_layers[0].shape



      exit()
      c_embedding = sess.run(embedding, feed_dict={image_u:batch_cimages})
      d_embedding = sess.run(embedding, feed_dict={image_u:batch_dimages})
      c_embedding = np.squeeze(c_embedding)
      d_embedding = np.squeeze(d_embedding)

      # now that we have the embeddings, interpolate between them
      alpha = np.linspace(0,1, num=5)
      latent_vectors = []
      x1 = c_embedding
      x2 = d_embedding
      for a in alpha:
         vector = x1*(1-a) + x2*a
         latent_vectors.append(vector)
      latent_vectors = np.asarray(latent_vectors)

      cc = 0
      for e in latent_vectors:
         e = np.random.normal(-1.5e-6, 7e-6, size=[1,1,1,512]).astype(np.float32)
         int_img = np.squeeze(sess.run(gen_image2, feed_dict={embedding_p:e, image_u:batch_cimages}))
         misc.imsave('./'+str(cc)+'.png', int_img)
         cc+=1
      exit()
