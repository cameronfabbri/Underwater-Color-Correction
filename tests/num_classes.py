import glob
import ntpath

image_files = glob.glob('test_images/ugan_1.0/*.png')

classes = []

for i in image_files:
   i = ntpath.basename(i)
   img = i.split('_')[0]

   classes.append(img)


print len(set(classes))

