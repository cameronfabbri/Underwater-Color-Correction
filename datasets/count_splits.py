import glob
import ntpath

images = glob.glob('test/*.jpg')

classes = []

for i in images:
   i = ntpath.basename(i).split('.')[0].split('_')[0]
   classes.append(i)


print set(classes)


