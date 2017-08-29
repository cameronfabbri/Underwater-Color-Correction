from shutil import move

import os
import glob


test = glob.glob('testB/*.jpg')

for t in test:

   t = t.replace('test','train')
   os.remove(t)
   #t = t.split('/')[-1]

   #src  = 'trainB/'+t
   #dest = 'testB/'+t

   #move(src, dest)
