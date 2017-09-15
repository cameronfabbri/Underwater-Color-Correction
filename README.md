Color correction for underwater vision.

CycleGAN was used to perform style transfer on 'normal' (above water) Imagenet photos to appear to
be underwater. I chose maybe 5 imagenet categories and manually separated them into obviously underwater
with bad colors and hues etc, and the other photos with good colors.

Three diving videos will be considered for Jahid's stuff
diving1: https://www.youtube.com/watch?v=QmRFmhILd5o

diving2: https://www.youtube.com/watch?v=OSdrb1XNXZI

diving3: https://www.youtube.com/watch?v=ALN6y2PLCq0

`tests` is a directory where I am performing various tests

`test_images` is a folder with ALL 1817x4 testing images.
There are four sets of tests images:
   - original images
   - cyclegan images
   - ugan images with IG = 1.0
   - ugan images with IG = 0.0


test
