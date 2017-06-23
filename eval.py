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
   
   global_step = tf.Variable(0, name='global_step', trainable=False)

   EXPERIMENT_DIR = 'checkpoints/'
   IMAGES_DIR     = EXPERIMENT_DIR+'images/'

   BATCH_SIZE=4
   Data = data_ops.loadData(BATCH_SIZE, train=False)
   # underwater grayscale image
   ugray = Data.ugray
   # underwater ab channels
   uab   = Data.uab

   # generated corrected colors
   gen_ab = pix2pix.netG(ugray, uab)
   
   new_u = data_ops.augment(gen_ab, ugray)
   new_u = tf.image.convert_image_dtype(new_u, dtype=tf.uint8, saturate=True)

   og_u = data_ops.augment(uab, ugray)
   og_u = tf.image.convert_image_dtype(og_u, dtype=tf.uint8, saturate=True)

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

   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   merged_summary_op = tf.summary.merge_all()

   old, new = sess.run([og_u, new_u])

   i = 0
   for img in old:
      misc.imsave(IMAGES_DIR+str(step)+'_'+str(i)+'_old.png', img)
      i += 1
   i = 0
   for img in new:
      misc.imsave(IMAGES_DIR+str(step)+'_'+str(i)+'_new.png', img)
      i += 1

   coord.request_stop()
   coord.join(threads)
