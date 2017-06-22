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
   parser.add_argument('--EPOCHS',        required=False,default=4,type=int,help='Number of epochs for GAN')
   parser.add_argument('--BATCH_SIZE',    required=False,default=4,type=int,help='Batch size')
   parser.add_argument('--LOSS_METHOD',   required=False,default='gan',help='Loss function for GAN')
   parser.add_argument('--LEARNING_RATE', required=False,default=1e-4,type=float,help='Learning rate')
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

   # first load the data
   #utrain_paths, utest_paths, coco_paths = load()

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # Old data loading stuff that doesn't work because of dumb tensorflow queue messups
   # load data
   Data = data_ops.loadData(BATCH_SIZE)
   # underwater grayscale image
   ugray = Data.ugray
   # underwater ab channels
   uab   = Data.uab
   # coco gray
   coco_L  = Data.coco_L
   # coco ab channels
   coco_ab = Data.coco_ab
   # want to send underwater LAB to G and have it predict new AB
   x = tf.concat([ugray, uab], axis=3)
   '''
      
   u_img = tf.placeholder(tf.float32, shape=[256,256,3],name='underwater_L')
   u_lab = data_ops.rgb_to_lab(u_img)
   uL_chan, ua_chan, ub_chan = data_ops.preprocess_lab(u_lab)
   ugray_images = tf.expand_dims(uL_chan, axis=2)   # shape (?,?,1)
   uab_images = tf.stack([ua_chan, ub_chan], axis=2) # shape (?,?,2)

   print ugray_images
   print uab_images
   exit()

   # underwater images
   #underwater_L  = tf.placeholder(tf.float32, shape=[BATCH_SIZE,256,256,1],name='underwater_L')
   #underwater_ab = tf.placeholder(tf.float32, shape=[BATCH_SIZE,256,256,2],name='underwater_ab')

   # above water images
   #coco_L  = tf.placeholder(tf.float32, shape=[BATCH_SIZE,256,256,1],name='coco_L')
   #coco_ab = tf.placeholder(tf.float32, shape=[BATCH_SIZE,256,256,2],name='coco_ab')
   '''

   # generated corrected colors
   #gen_ab = pix2pix.netG(underwater_L, underwater_ab)
   gen_ab = pix2pix.netG(ugray, uab)

   # send coco images to D
   D_real = pix2pix.netD(coco_L, coco_ab)

   # send corrected underwater images to D
   #D_fake = pix2pix.netD(underwater_L, gen_ab, reuse=True)
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
   #try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   #except:pass
   #try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   #except:pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]
      
   G_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)

   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
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


   coord.request_stop()
   coord.join(threads)
