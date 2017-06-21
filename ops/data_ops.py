'''

Operations used for data management

MASSIVE help from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

'''

from __future__ import division
from __future__ import absolute_import

from scipy import misc
from skimage import color
import collections
import tensorflow as tf
import numpy as np
import math
import time
import random
import glob
import os
import fnmatch
import cPickle as pickle

Data = collections.namedtuple('trainData', 'ugray, uab, coco')

def preprocess(image):
    with tf.name_scope('preprocess'):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope('deprocess'):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope('preprocess_lab'):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope('deprocess_lab'):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
        #return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=2)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb



def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope('rgb_to_lab'):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        with tf.name_scope('srgb_to_xyz'):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope('xyz_to_cielab'):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope('lab_to_rgb'):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])
        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope('cielab_to_xyz'):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope('xyz_to_srgb'):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

def getPaths(data_dir, gray_images=None, ext='png'):
   pattern   = '*.'+ext
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fname_ = os.path.join(d,filename)
            if gray_images is not None:
               if fname_ in gray_images:
                  continue
            image_paths.append(fname_)
   return image_paths


# TODO add in files to exclude (gray ones)
def loadData(batch_size, train=True):

   # pickle files containing images of underwater
   pkl_utrain_file = 'files/underwater_train.pkl'
   pkl_utest_file  = 'files/underwater_test.pkl'

   # microsoft coco images
   pkl_coco_file = 'files/coco_train.pkl'

   # load saved pickle files if already have them
   if os.path.isfile(pkl_utrain_file) and os.path.isfile(pkl_utest_file) and os.path.isfile(pkl_coco_file):
      print 'Found pickle files'
      utrain_paths     = pickle.load(open(pkl_utrain_file, 'rb'))
      utest_paths      = pickle.load(open(pkl_utest_file, 'rb'))
      coco_train_paths = pickle.load(open(pkl_coco_file, 'rb'))
   else:
      print 'Reading data...'
      utrain_dir  = '/mnt/data2/images/underwater/youtube/'
      coco_dir    = '/mnt/data2/images/coco/'

      # get all paths for underwater images
      u_paths    = getPaths(utrain_dir)
      # get all paths for coco images
      coco_paths = getPaths(coco_dir)

      # shuffle all the underwater paths before splitting into test/train
      random.shuffle(u_paths)

      # shuffle coco images because why not
      random.shuffle(coco_paths)

      # take 90% for train, 10% for test
      train_num = int(0.9*len(u_paths))
      utrain_paths = u_paths[:train_num]
      utest_paths  = u_paths[train_num:]

      # write train underwater data
      pf   = open(pkl_utrain_file, 'wb')
      data = pickle.dumps(utrain_paths)
      pf.write(data)
      pf.close()

      # write test underwater data
      pf   = open(pkl_utest_file, 'wb')
      data = pickle.dumps(utest_paths)
      pf.write(data)
      pf.close()

      # write test underwater data
      pf   = open(pkl_coco_file, 'wb')
      data = pickle.dumps(coco_paths)
      pf.write(data)
      pf.close()


   decode = tf.image.decode_image

   coco_paths = utrain_paths

   # load underwater images
   with tf.name_scope('load_underwater'):
      path_queue = tf.train.string_input_producer(utrain_paths, shuffle=True)
      reader = tf.WholeFileReader()
      paths, contents = reader.read(path_queue)
      raw_input = decode(contents)
      raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

      assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message='image does not have 3 channels')
      with tf.control_dependencies([assertion]):
         raw_input = tf.identity(raw_input)

      raw_input.set_shape([None, None, 3])

      # randomly flip image
      seed = random.randint(0, 2**31 - 1) 
      raw_input = tf.image.random_flip_left_right(raw_input, seed=seed)
      
      # convert to LAB and process gray channel and color channels
      lab = rgb_to_lab(raw_input)
      L_chan, a_chan, b_chan = preprocess_lab(lab)
      gray_images = tf.expand_dims(L_chan, axis=2)   # shape (?,?,1)
      ab_images = tf.stack([a_chan, b_chan], axis=2) # shape (?,?,2)
      
      gray_images = tf.image.resize_images(gray_images, [256, 256], method=tf.image.ResizeMethod.AREA)
      ab_images   = tf.image.resize_images(ab_images, [256, 256], method=tf.image.ResizeMethod.AREA)

      u_paths_batch, gray_batch, ab_batch = tf.train.batch([paths, gray_images, ab_images], batch_size=batch_size)
      
   # load the coco images
   with tf.name_scope('load_coco'):
      path_queue = tf.train.string_input_producer(coco_paths, shuffle=True)
      reader = tf.WholeFileReader()
      paths, contents = reader.read(path_queue)
      raw_input = decode(contents)
      raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

      assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message='image does not have 3 channels')
      with tf.control_dependencies([assertion]):
         raw_input = tf.identity(raw_input)

      raw_input.set_shape([None, None, 3])

      # randomly flip image
      seed = random.randint(0, 2**31 - 1) 
      raw_input = tf.image.random_flip_left_right(raw_input, seed=seed)
      
      # convert to LAB. Don't need them separated though
      coco_images = rgb_to_lab(raw_input)
      coco_images = tf.image.resize_images(coco_images, [256,256], method=tf.image.ResizeMethod.AREA)

      coco_paths_batch, coco_batch = tf.train.batch([paths, coco_images], batch_size=batch_size)
      

   # need to return underwater L, a, b. coco LAB

   return Data(
      ugray=gray_batch,
      uab=ab_batch,
      coco=coco_batch,
   )
