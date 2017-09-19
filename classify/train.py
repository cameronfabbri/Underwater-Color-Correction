import cPickle as pickle
import tensorflow as tf
from tqdm import tqdm
from scipy import misc
import numpy as np
import argparse
import ntpath
import sys
import os
import time
import glob

sys.path.insert(0, '../ops/')
from tf_ops import *

import data_ops

from network import *

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE',    required=False,default=8,type=int,help='Batch size')
   parser.add_argument('--LEARNING_RATE', required=False,default=1e-4,type=float,help='Learning rate')
   a = parser.parse_args()

   LEARNING_RATE = float(a.LEARNING_RATE)
   BATCH_SIZE    = a.BATCH_SIZE
   
   EXPERIMENT_DIR  = 'checkpoints/'
   
   print
   print 'Creating',EXPERIMENT_DIR
   try: os.makedirs(EXPERIMENT_DIR)
   except: pass
   
   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 227, 227, 3), name='images')
   labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2), name='labels')

   logits = network(images, num_classes=2, is_training=True)

   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
   
   # tensorboard summaries
   try: tf.summary.scalar('loss', tf.reduce_mean(loss))
   except: pass

   train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

   saver = tf.train.Saver(max_to_keep=2)

   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

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

   merged_summary_op = tf.summary.merge_all()

   # underwater photos
   trainA_paths  = np.asarray(glob.glob('../datasets/underwater_imagenet/trainA/*.jpg'))
   trainA_labels = np.repeat(1, len(trainA_paths))

   # normal photos (ground truth)
   trainB_paths  = np.asarray(glob.glob('../datasets/underwater_imagenet/trainB/*.jpg'))
   trainB_labels = np.repeat(0, len(trainB_paths))

   train_paths  = np.concatenate([trainA_paths, trainB_paths])
   train_labels = np.concatenate([trainA_labels, trainB_labels])
   
   num_train = len(train_paths)

   while True:

      epoch_num = step/(num_train/BATCH_SIZE)

      idx = np.random.choice(np.arange(num_train), BATCH_SIZE, replace=False)
      batch_paths = train_paths[idx]
      batch_ls    = train_labels[idx]
      
      batch_images = np.empty((BATCH_SIZE, 227, 227, 3), dtype=np.float32)
      batch_labels = np.empty((BATCH_SIZE, 2), dtype=np.float32)

      i = 0
      for image,label in zip(batch_paths, batch_ls):
         zero = np.asarray([0,0])

         img = misc.imread(image).astype('float32')
         img = misc.imresize(img, (227, 227))

         zero[label] = 1
         batch_labels[i, ...] = zero
         batch_images[i, ...] = img

         #print batch_labels[0]
         #misc.imsave('img_'+str(i)+'.png', img)
         #exit()
         i += 1

      print batch_labels
      exit()

      sess.run(train_op, feed_dict={images:batch_images, labels:batch_labels})
      log_, loss_, summary = sess.run([logits, loss, merged_summary_op], feed_dict={images:batch_images, labels:batch_labels})
      
      summary_writer.add_summary(summary, step)
      print 'epoch:',epoch_num,'step:',step,'loss:',loss_,log_[0]
      step += 1
      
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'




