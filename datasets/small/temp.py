import os
import glob


test = glob.glob('testA/*.jpg')

for t in test:

   t = t.split('/')[-1]

   src  = 'trainB/'+t
   dest = 'testB/'+t

   os.rename(src, dest)
