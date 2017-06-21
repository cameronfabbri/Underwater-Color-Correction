'''

Cameron Fabbri
Script to find gray images in a folder

'''

import scipy.misc as misc
from tqdm import tqdm
import fnmatch
import sys
import os

f = open('gray_images_standard.txt', 'a')

if __name__ == '__main__':

   data_dir = sys.argv[1]

   pattern = '*.jpg'
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_paths.append(os.path.join(d,filename))

   for image in tqdm(image_paths):

      img = misc.imread(image)
      shape = len(img.shape)

      if shape < 3:
         print image
         f.write(image+'\n')
