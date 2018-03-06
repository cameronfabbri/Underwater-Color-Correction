import os
import glob

traina = glob.glob('trainA/*.jpg')
trainb_ = glob.glob('trainB/*.jpg')
import ntpath

trainb = []
for b in trainb_:
   trainb.append(ntpath.basename(b))

for a in traina:
   a = ntpath.basename(a)
   if a not in trainb:
      print a
      os.remove('trainA/'+a)


