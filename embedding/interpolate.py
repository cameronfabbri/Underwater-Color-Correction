import numpy as np
import scipy.misc as misc 
      
#x1 = misc.imread('clean/turtle1.jpg')
#x2 = misc.imread('distorted/turtle1.jpg')
x1 = misc.imread('n01496331_1692_real.png')
x2 = misc.imread('n01496331_1692_gen.png')

num_int = 10

alpha = np.linspace(0,1, num=num_int)
latent_vectors = []
for a in alpha:
   vector = x1*(1-a) + x2*a
   latent_vectors.append(vector)
latent_vectors = np.asarray(latent_vectors)

canvas = 255*np.ones((256, (num_int)*256, 3), dtype=np.uint8)

i = 0
start=0
end=256
for e in latent_vectors:
   #misc.imsave(str(i)+'.png', e)
   #i += 1
   canvas[:,start:end,:] = e
   start=end
   end = start+256
misc.imsave('canvas.png', canvas)
