import os
import time
from argparse import ArgumentParser

import numpy as np
import torch

from fast_optics_correction import OpticsCorrection
from fast_optics_correction import utils, filters


## Parameters
def Options():
    parser = ArgumentParser()

    # Paths parameters
    dirname = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--name', type=str)
    parser.add_argument('--imfolder', default=os.path.join(dirname, 'pictures'), type=str)
    parser.add_argument('--savefolder', default=os.path.join(dirname, 'results'), type=str)

    # Patch decomposition parameters
    parser.add_argument('--patch_size', default=400, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--overlap_percentage', default=0.25, type=float)

    # Polyblur parameters
    parser.add_argument('--polyblur_iteration', default=3, type=int)
    parser.add_argument('--do_decomposition', default=False, action='store_true')
    parser.add_argument('--do_edgetaper', default=False, action='store_true')
    parser.add_argument('--do_halo_removal', default=False, action='store_true')

    # Blur estimation parameters
    parser.add_argument('--c', default=0.416, type=float)
    parser.add_argument('--sigma_b', default=0.358, type=float)
    parser.add_argument('--q', default=0.0001, type=float)
    parser.add_argument('--ker_size', default=25, type=int)

    # Deconvolution parameters
    parser.add_argument('--alpha', default=6, type=float)
    parser.add_argument('--b', default=1, type=float)

    # Domain transform parameters
    parser.add_argument('--sigma_s', default=200, type=float)
    parser.add_argument('--sigma_r', default=0.1, type=float)

    return parser


opts = Options()
args = opts.parse_args()

device = torch.device('cuda:0')
print('Will run on', device)


## Read the image
name_ = args.name
name = name_[:-4]
impath = os.path.join(args.imfolder, '%s' % name_)
img = utils.read_image(impath)

print('Image to be restored: %s (%d x %d)' % (name, img.shape[1], img.shape[0]))

## Print the arguments
print()
print('###### Parameters #####')
print('Patch decomposition parameters')
print('  --patch_size:          %d' % args.patch_size)
print('  --batch_size:          %d' % args.batch_size)
print('  --overlap_percentage:  %1.2f' % args.overlap_percentage)
print('Polyblur parameters')
print('  --polyblur_iterations: %d' % args.polyblur_iteration)
print('  --do_decomposition:    %s' % args.do_decomposition)
print('  --do_edgetaper:        %s' % args.do_edgetaper)
print('  --do_halo_removal:     %s' % args.do_halo_removal)
print('Blur estimation parameters')
print('  --c:                   %1.3f' % args.c)
print('  --sigma_b:             %1.3f' % args.sigma_b)
print('  --q:                   %1.3f' % args.q)
print('  --ker_size:            %1.3f' % args.ker_size)
print('Deconvolution parameters')
print('  --alpha:               %1.3f' % args.alpha)
print('  --b:                   %1.3f' % args.b)
print('Domain transform parameters')
print('  --sigma_s:             %3.2f' % args.sigma_s)
print('  --sigma_r:             %1.2f' % args.sigma_r)
print()

## Load the model
model = OpticsCorrection(patch_size=args.patch_size, overlap_percentage=args.overlap_percentage,
                         ker_size=args.ker_size, batch_size=args.batch_size)
model = model.to(device)


## CUDA warmup -- to not bias running time
torch.fft.fft2(torch.randn(60,3,400,400, device=device))
with torch.no_grad():
    model.defringer(torch.randn(60,2,400,400, device=device))
filters.extract_tiles(torch.randn(10,3,400,400, device=device), (3,3))
filters.recursive_filter(torch.randn(10,3,400,400, device=device))
torch.randn(10,3,400,400, device=device).cpu()


## Inference
img = utils.to_tensor(img).unsqueeze(0)  # Define the array on the CPU! (important for larger images)
img = img.to(device)
tic = time.time()
with torch.no_grad():
    img_corrected = model(img, sigma_b=args.sigma_b, c=args.c, q=args.q,
                          polyblur_iteration=args.polyblur_iteration, alpha=args.alpha, b=args.b,
                          do_decomposition=args.do_decomposition, 
                          do_halo_removal=args.do_halo_removal,
                          do_edgetaper=args.do_edgetaper, sigma_s=args.sigma_s, sigma_r=args.sigma_r)
tac = time.time()
print('Restoration took  %1.3f seconds.' % (tac - tic))
img_corrected = utils.to_array(img_corrected.cpu())
img = utils.to_array(img.cpu())


## Gamma curve as simple ISP
img = np.clip(img, a_min=1e-8, a_max=1.0) ** (1./2.2)
img_corrected = np.clip(img_corrected, a_min=1e-8, a_max=1.0) ** (1./2.2)


## Saving the images
os.makedirs(args.savefolder, exist_ok=True)

print('Will save that images at %s' % args.savefolder)

utils.write_image(os.path.join(args.savefolder, '%s_original.png' % name), img)
utils.write_image(os.path.join(args.savefolder, '%s_corrected.png' % name), img_corrected)

