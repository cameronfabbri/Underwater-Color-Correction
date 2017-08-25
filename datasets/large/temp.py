from shutil import copyfile


import os
import glob


test = glob.glob('testA/*.jpg')

for t in test:

   t = t.split('/')[-1]

   src  = 'trainB/'+t
   dest = 'testB/'+t

   copyfile(src, dest)
