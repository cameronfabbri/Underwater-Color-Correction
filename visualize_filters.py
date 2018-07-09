import tensorflow as tf
import sys
import cPickle as pickle
import numpy as np
from math import sqrt
import scipy.misc as misc

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *

def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]
  print 'Y:',Y
  print 'X:',X
  print 'channels:',channels

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x


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

   EXPERIMENT_DIR  = 'checkpoints/LOSS_METHOD_'+LOSS_METHOD\
                     +'/NETWORK_'+NETWORK\
                     +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                     +'/IG_WEIGHT_'+str(IG_WEIGHT)\
                     +'/AUGMENT_'+str(AUGMENT)\
                     +'/DATA_'+DATA+'/'\

   if NETWORK == 'pix2pix': from pix2pix import *
   if NETWORK == 'resnet':  from resnet import *

   image_u = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name='image_u')
   layers    = netG_encoder(image_u)
   gen_image = netG_decoder(layers)   

   saver = tf.train.Saver()

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


   with tf.variable_scope('g_enc_conv1'):
      tf.get_variable_scope().reuse_variables()
      weights = tf.get_variable('weights')
      grid = put_kernels_on_grid (weights)
      g = np.squeeze(sess.run(grid))
      misc.imsave('filters_enc_conv1.png', g)
