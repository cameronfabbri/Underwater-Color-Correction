import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys

sys.path.insert(0, 'ops/')
from tf_ops import lrelu, conv2d, batch_norm, conv2d_transpose, relu, tanh

def netG(underwater_L, underwater_ab):
  
   x = tf.concat([underwater_L, underwater_ab], axis=3)

   with tf.variable_scope('g_enc1'): enc_conv1 = lrelu(conv2d(x, 64, stride=2))
   with tf.variable_scope('g_enc2'): enc_conv2 = lrelu(batch_norm(conv2d(enc_conv1, 128, stride=2, kernel_size=4)))
   with tf.variable_scope('g_enc3'): enc_conv3 = lrelu(batch_norm(conv2d(enc_conv2, 256, stride=2, kernel_size=4)))
   with tf.variable_scope('g_enc4'): enc_conv4 = lrelu(batch_norm(conv2d(enc_conv3, 512, stride=2, kernel_size=4)))
   with tf.variable_scope('g_enc5'): enc_conv5 = lrelu(batch_norm(conv2d(enc_conv4, 512, stride=2, kernel_size=4)))
   with tf.variable_scope('g_enc6'): enc_conv6 = lrelu(batch_norm(conv2d(enc_conv5, 512, stride=2, kernel_size=4)))
   with tf.variable_scope('g_enc7'): enc_conv7 = lrelu(batch_norm(conv2d(enc_conv6, 512, stride=2, kernel_size=4)))
   with tf.variable_scope('g_enc8'): enc_conv8 = lrelu(batch_norm(conv2d(enc_conv7, 512, stride=2, kernel_size=4)))

   print 'enc_conv1:',enc_conv1
   print 'enc_conv2:',enc_conv2
   print 'enc_conv3:',enc_conv3
   print 'enc_conv4:',enc_conv4
   print 'enc_conv5:',enc_conv5
   print 'enc_conv6:',enc_conv6
   print 'enc_conv7:',enc_conv7
   print 'enc_conv8:',enc_conv8

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


def netD(L_images, ab_images, reuse=False):
   print
   print 'netD'
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      input_images = tf.concat([L_images, ab_images], axis=3)

      with tf.variable_scope('d_conv1'): conv1 = lrelu(conv2d(input_images, 64, kernel_size=4, stride=2))
      with tf.variable_scope('d_conv2'): conv2 = lrelu(batch_norm(conv2d(conv1, 128, kernel_size=4, stride=2)))
      with tf.variable_scope('d_conv3'): conv3 = lrelu(batch_norm(conv2d(conv2, 256, kernel_size=4, stride=2)))
      with tf.variable_scope('d_conv4'): conv4 = lrelu(batch_norm(conv2d(conv3, 512, kernel_size=4, stride=1)))
      with tf.variable_scope('d_conv5'): conv5 = conv2d(conv4, 1, stride=1)

      print conv1
      print conv2
      print conv3
      print conv4
      print conv5
      return conv5


def netD_ab(ab_images, num_gpu, reuse=False):
   ndf = 64
   n_layers = 3
   layers = []
   print
   print 'netD'
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      if num_gpu == 0: gpus = ['/cpu:0']
      elif num_gpu == 1: gpus = ['/gpu:0']
      elif num_gpu == 2: gpus = ['/gpu:0', '/gpu:1']
      elif num_gpu == 3: gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
      elif num_gpu == 4: gpus = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

      for d in gpus:
         with tf.device(d):

            input_images = ab_images

            with tf.variable_scope('d_ab_conv1'): conv1 = lrelu(conv2d(input_images, 64, kernel_size=4, stride=2))
            with tf.variable_scope('d_ab_conv2'): conv2 = lrelu(batch_norm(conv2d(conv1, 128, kernel_size=4, stride=2)))
            with tf.variable_scope('d_ab_conv3'): conv3 = lrelu(batch_norm(conv2d(conv2, 256, kernel_size=4, stride=2)))
            with tf.variable_scope('d_ab_conv4'): conv4 = lrelu(batch_norm(conv2d(conv3, 512, kernel_size=4, stride=1)))
            with tf.variable_scope('d_ab_conv5'): conv5 = conv2d(conv4, 1, stride=1)

            print conv1
            print conv2
            print conv3
            print conv4
            print conv5
            return conv5



'''
'''
def energyEncoder(ab_images, reuse=False):
   print 'DISCRIMINATOR' 
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = layers.conv2d(ab_images, 64, 4, stride=2, activation_fn=None, scope='d_conv1')
      conv1 = lrelu(conv1)
      print 'conv1:',conv1

      conv2 = layers.conv2d(conv1, 128, 4, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv2')
      conv2 = lrelu(conv2)
      print 'conv2:',conv2
      
      conv3 = layers.conv2d(conv2, 256, 4, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv3')
      conv3 = lrelu(conv3)
      print 'conv3:',conv3
      
      conv4 = layers.conv2d(conv3, 512, 4, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv4')
      conv4 = lrelu(conv4)
      print 'conv4:',conv4
      
      tf.add_to_collection('vars', conv1)
      tf.add_to_collection('vars', conv2)
      tf.add_to_collection('vars', conv3)
      tf.add_to_collection('vars', conv4)

      return conv4

def energyDecoder(encoded, reuse=False):
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      
      conv5 = layers.conv2d_transpose(encoded, 256, 4, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv5')
      conv5 = lrelu(conv5)

      conv6 = layers.conv2d_transpose(conv5, 128, 4, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv6')
      conv6 = lrelu(conv6)
      
      conv7 = layers.conv2d_transpose(conv6, 64, 4, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv7')
      conv7 = lrelu(conv7)

      conv8 = layers.conv2d_transpose(conv7, 2, 4, stride=2, activation_fn=tf.nn.tanh, scope='d_conv8')

      print 'encoded:',encoded
      print 'conv5:',conv5
      print 'conv6:',conv6
      print 'conv7:',conv7
      print 'conv8:',conv8
      
      print 'END D\n'
      tf.add_to_collection('vars', conv5)
      tf.add_to_collection('vars', conv6)
      tf.add_to_collection('vars', conv7)
      tf.add_to_collection('vars', conv8)
      return conv8


'''
   Only encoding and decoding the ab values, then concatenating onto the
   lightness channel and taking the mse
'''
def energyNetD(L_images, ab_images, batch_size, reuse=False):
   # concat lightness channel with ab values
   input_images = tf.concat([L_images, ab_images], axis=3)

   # encode the ab values
   encoded = energyEncoder(ab_images, reuse=reuse)

   # decode ab values
   decoded = energyDecoder(encoded, reuse=reuse)
   decoded_images = tf.concat([L_images, decoded], axis=3)
   return mse(decoded_images, input_images, batch_size), encoded, decoded

def mse(pred, real, batch_size):
   return tf.sqrt(2*tf.nn.l2_loss(pred-real))/batch_size

