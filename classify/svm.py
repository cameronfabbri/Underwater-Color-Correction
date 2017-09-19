import numpy as np
from sklearn.svm import SVC
import scipy.misc as misc
import glob
import os
import fnmatch
import random
from tqdm import tqdm
import time

if __name__ == '__main__':

   # distorted
   trainA_paths  = np.asarray(glob.glob('../datasets/underwater_imagenet/trainA/*.jpg'))

   # non distorted
   trainB_paths  = np.asarray(glob.glob('../datasets/underwater_imagenet/trainB/*.jpg'))

   images = []
   labels = []

   for path in trainA_paths:
      img = misc.imread(path)/255.0
      img = misc.imresize(img, (64,64))
      img = np.reshape(img, (64*64*3))
      images.append(img)
      labels.append(1)

   for path in trainB_paths:
      img = misc.imread(path)/255.0
      img = misc.imresize(img, (64,64))
      img = np.reshape(img, (64*64*3))
      images.append(img)
      labels.append(0)

   images = np.asarray(images)
   labels = np.asarray(labels)

   print images.shape
   print labels.shape

   print 'Fitting...'
   s = time.time()
   clf = SVC()
   clf.fit(images, labels)
   print 'Done',time.time()-s

   test_paths = []
   image_list = []
   data_dir = '/mnt/data2/cyclegan/imagenet'
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, '*.JPEG'):
            test_paths.append(os.path.join(d,filename))
   
   random.shuffle(test_paths)

   f1 = open('distorted.txt', 'a')
   f2 = open('nondistorted.txt', 'a')

   for path in tqdm(test_paths):

      try:
         img = misc.imread(path)/255.0
         img = misc.imresize(img, (64,64))
         img = np.reshape(img, (64*64*3))
      except:
         continue

      p = clf.predict(img.reshape(1, -1))[0]

      if p == 0: f2.write(path+'\n')
      else: f1.write(path+'\n')


