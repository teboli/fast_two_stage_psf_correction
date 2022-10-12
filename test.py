import os
import time

import numpy as np
import torch

from fast_optics_correction import OpticsCorrection
from fast_optics_correction import utils


## Parameters
c = 0.416
sigma_b = 0.358
q = 0.0001
patch_size = 400
overlap_percentage = 0.25
ker_size = 25
polyblur_iteration = 3  # can be also 2 or 3
# alpha = 2; b = 4
# alpha = 1; b = 6
# alpha = 3; b = 6
alpha = 6; b = 1
batch_size = 60
do_decomposition = False  # should we do a base/detail decomp. for not enhancing noise and artifacts?
do_edgetaper = True
do_halo_removal = True

device = torch.device('cuda:1')
print('Will run on', device)

## Read the image
# name = 'facade'; impath = './pictures/facade.jpg'
name = 'bridge'; impath = './pictures/bridge.jpg'
img = utils.read_image(impath)

print(name, impath)

## Load the model
model = OpticsCorrection(patch_size=patch_size, overlap_percentage=overlap_percentage,
                         ker_size=ker_size, batch_size=batch_size)
model = model.to(device)


## CUDA warmup -- to not bias running time
torch.fft.fft2(torch.randn(60,3,400,400, device=device))
model.defringer(torch.randn(10,2,400,400, device=device))

## Inference
img = utils.to_tensor(img).unsqueeze(0).to(device)
tic = time.time()
img_corrected = model(img, sigma_b=sigma_b, c=c, q=q,
                      polyblur_iteration=polyblur_iteration, alpha=alpha, b=b,
                      do_decomposition=do_decomposition, 
                      do_halo_removal=do_halo_removal,
                      do_edgetaper=do_edgetaper)
tac = time.time()
img_corrected = utils.to_array(img_corrected.cpu())
img = utils.to_array(img.cpu())
print('Restoration took %2.2f seconds.' % (tac - tic))


## Gamma curve as simple ISP
img = np.clip(img, a_min=1e-8, a_max=1.0) ** (1./2.2)
img_corrected = np.clip(img_corrected, a_min=1e-8, a_max=1.0) ** (1./2.2)


## Saving the images
savefolder = './results/'
os.makedirs(savefolder, exist_ok=True)


utils.write_image(os.path.join(savefolder, '%s_original.png' % name), img)
utils.write_image(os.path.join(savefolder, '%s_corrected.png' % name), img_corrected)

