import numpy as np
import scipy.misc as misc 
      
x1 = misc.imread('clean/turtle1.jpg')
x2 = misc.imread('distorted/turtle1.jpg')

num_int = 10

alpha = np.linspace(0,1, num=num_int)
latent_vectors = []
for a in alpha:
   vector = x1*(1-a) + x2*a
   latent_vectors.append(vector)
latent_vectors = np.asarray(latent_vectors)

i = 0
for e in latent_vectors:
   misc.imsave(str(i)+'.png', e)
   i += 1
