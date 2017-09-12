import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import scipy.misc as misc
import scipy.stats.signaltonoise as snr



real_img  = misc.imread('n01496331_11938_real.png')
cycle_gan = misc.imread('n01496331_11938.png')
ours      = misc.imread('n01496331_11938_gen_IG_0.0.png')
'''
print ssim(real_img, cycle_gan, multichannel=True)
print ssim(real_img, ours, multichannel=True)
print
print psnr(real_img, cycle_gan)
print psnr(real_img, ours)
'''

#print np.linalg.norm(np.fft.fftn(real_img))
#print np.linalg.norm(np.fft.fftn(cycle_gan))
#print np.linalg.norm(np.fft.fftn(ours))

print snr(cycle_gan)

