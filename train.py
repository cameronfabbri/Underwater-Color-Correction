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

sys.path.insert(0, 'ops/')
from tf_ops import *

import data_ops

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--EPOCHS',        required=True,type=int,help='Number of epochs for GAN')
   parser.add_argument('--BATCH_SIZE',    required=False,type=int,default=32,help='Batch size')
   parser.add_argument('--LOSS_METHOD',   required=False,default='wasserstein',help='Loss function for GAN')
   parser.add_argument('--LEARNING_RATE', required=False,type=float,default=1e-4,help='Learning rate')
   a = parser.parse_args()

   LEARNING_RATE = a.LEARNING_RATE
   LOSS_METHOD   = a.LOSS_METHOD
   BATCH_SIZE    = a.BATCH_SIZE
   EPOCHS        = a.EPOCHS
   
   EXPERIMENT_DIR = 'checkpoints/'
   IMAGES_DIR     = EXPERIMENT_DIR+'images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir(EXPERIMENT_DIR)
   except: pass
   try: os.mkdir(IMAGES_DIR)
   except: pass
   
   # write all this info to a pickle file in the experiments directory
   exp_info = dict()
   exp_info['EPOCHS']        = EPOCHS
   exp_info['BATCH_SIZE']      = BATCH_SIZE
   exp_info['LOSS_METHOD']   = LOSS_METHOD
   exp_info['LEARNING_RATE'] = LEARNING_RATE
   exp_pkl = open(EXPERIMENT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()
   
   print
   print 'EPOCHS:        ',EPOCHS
   print 'BATCH_SIZE:    ',BATCH_SIZE
   print 'LOSS_METHOD:   ',LOSS_METHOD
   print 'LEARNING_RATE: ',LEARNING_RATE
   print

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # load data
   Data = data_ops.loadData(BATCH_SIZE)

   # underwater grayscale image
   ugray = Data.ugray

   # underwater ab channels
   uab   = Data.uab

   # full color image of coco
   coco  = Data.coco

   # want to send underwater LAB to G and have it predict new AB
   gen_ab = pix2pix.netG(ugray, uab)

   # send coco images to D
   D_real = pix2pix.netD(coco)

   # send corrected underwater images to D
   D_fake = pix2pix.netD(ugray, gen_ab, reuse=True)

   e = 1e-12
   if LOSS_METHOD == 'least_squares':
      print 'Using least squares loss'
      errD_real = tf.nn.sigmoid(D_real)
      errD_fake = tf.nn.sigmoid(D_fake)
      errG = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
   if LOSS_METHOD == 'gan':
      print 'Using original GAN loss'
      D_real = tf.nn.sigmoid(D_real)
      D_fake = tf.nn.sigmoid(D_fake)
      errG = tf.reduce_mean(-tf.log(D_fake + e))
      errD = tf.reduce_mean(-(tf.log(D_real+e)+tf.log(1-D_fake+e)))
   
   # tensorboard summaries
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]
      
      
   G_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.initialize_all_variables()
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   step = sess.run(global_step)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)
   merged_summary_op = tf.summary.merge_all()
   start = time.time()
   while True:

      sess.run(D_train_op)
      sess.run(G_train_op)
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])

      summary_writer.add_summary(summary, step)
      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss
      step += 1
      
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'

