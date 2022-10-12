import torch
import torch.nn.functional as F
import torch.fft
import numpy as np
from skimage import filters

from . import utils
from .utils_fft import *


#####################################################################
######################## Convolution 2D #############################
#####################################################################

def convolve2d(img, kernel, ksize=25, padding='same', method='direct'):
    """
    A per kernel wrapper for torch.nn.functional.conv2d
    :param img: (B,C,H,W) torch.tensor, the input images
    :param kernel: (B,C,h,w) or 
                   (B,1,h,w) torch.tensor, the 2d blur kernels (valid for both deblurring methods), or 
                   [(B,C), (B,C), (B,C)] or 
                   [(B,1), (B,1), (B,1)], the separable 1d blur kernel parameters (valid only for spatial deblurring)
    :param padding: string, can be 'valid' or 'same' 
    : 
    :return imout: (B,C,H,W) torch.tensor, the filtered images
    """
    if method == 'direct':
        return conv2d_(img, kernel, padding)
    elif method == 'fft':
        X = torch.fft.fft2(utils.pad_with_kernel(img, kernel, mode='circular'))
        K = p2o(kernel, X.shape[-2:])
        return utils.crop_with_kernel( torch.real(torch.fft.ifft2(K * X)), kernel )
    else:
        raise('Convolution method %s is not implemented' % method)


def conv2d_(img, kernel, padding='same'):
    """
    Wrapper for F.conv2d with RGB images and kernels.
    """
    b, c, h, w = img.shape
    _, _, kh, kw = kernel.shape
    img = img.view(1, b*c, h, w)
    kernel = kernel.view(b*c, 1, kh, kw)
    return F.conv2d(img, kernel, groups=img.shape[1], padding=padding).view(b, c, h, w)


#####################################################################
####################### Bilateral filter ############################
#####################################################################


def extract_tiles(img, kernel_size, stride=1):
    b, c, _, _ = img.shape
    h, w = kernel_size
    tiles = F.unfold(img, kernel_size, stride)  # (B,C*H*W,L)
    tiles = tiles.permute(0, 2, 1)  # (B,L,C*H*W)
    tiles = tiles.view(b, -1, c, h ,w)
    return tiles


def bilateral_filter(I, ksize=5, sigma_spatial=1.0, sigma_color=0.1):
    ## precompute the spatial kernel: each entry of gw is a square spatial difference
    t = torch.arange(-ksize//2+1, ksize//2+1, device=I.device)
    xx, yy = torch.meshgrid(t, t, indexing='xy')
    gw = torch.exp(-(xx * xx + yy * yy) / (2 * sigma_spatial * sigma_spatial))  # (ksize, ksize)

    ## Create the padded array for computing the color shifts
    I_padded = utils.pad_with_kernel(I, ksize=ksize)

    ## Filtering
    var2_color = 2 * sigma_color * sigma_color
    return bilateral_filter_loop_(I, I_padded, gw, var2_color)


def bilateral_filter_loop_(I, I_padded, gw, var2, do_for=True):
    b, c, h, w = I.shape

    if do_for:  # memory-friendly option (Recommanded for larger images)
        J = torch.zeros_like(I)
        W = torch.zeros_like(I)
        for y in range(gw.shape[0]):
            yy = y + h
            # get the shifted image
            I_shifted = I_padded[..., y:yy, :]
            I_shifted = extract_tiles(I_shifted, kernel_size=(h,w), stride=1)  # (B,ksize,C,H,W)
            # color weight
            F = I_shifted - I.unsqueeze(1)  # (B,ksize,C,H,W)
            F = torch.exp(-F * F / var2) 
            # product with spatial weight
            F *= gw[y].view(-1, 1, 1, 1) # (B,ksize,C,H,W)
            J += torch.sum(F * I_shifted, dim=1)
            W += torch.sum(F, dim=1)
    else:  # pytorch-friendly option (Faster for smaller images and/or batch sizes)
        # get shifted images
        I_shifted = extract_tiles(I_padded, kernel_size=(h,w), stride=1)  # (B,ksize*ksize,C,H,W)
        F = I_shifted - I.unsqueeze(1)
        F = torch.exp( - F * F / var2)  # (B,ksize*ksize,C,H,W)
        # product with spatial weights
        F *= gw.view(-1, 1, 1, 1)
        J = torch.sum(F * I_shifted, dim=1)  # (B,C,H,W)
        W = torch.sum(F, dim=1)  # (B,C,H,W)
    return J / (W + 1e-5)

    


#####################################################################
###################### Classical filters ############################
#####################################################################


# @torch.jit.script
def fourier_gradients(images, freqs=None):
    """
    Compute the image gradients using Fourier interpolation as in Eq. (21a) and (21b)
        :param images: (B,C,H,W) torch.tensor
        :return grad_x, grad_y: tuple of 2 images of same dimensions as images that
                                are the vertical and horizontal gradients
    """
    ## Find fast size for FFT
    h, w = images.shape[-2:]
    h_fast, w_fast = images.shape[-2:]
    # h_fast = scipy.fft.next_fast_len(h)
    # w_fast = scipy.fft.next_fast_len(w)
    ## compute FT
    U = torch.fft.fft2(images)
    U = torch.fft.fftshift(U, dim=(-2, -1))
    ## Create the freqs components
    if freqs is None:
        freqh = (torch.arange(0, h_fast, device=images.device) - h_fast // 2).view(1,1,-1,1) / h_fast
        freqw = (torch.arange(0, w_fast, device=images.device) - w_fast // 2).view(1,1,1,-1) / w_fast
    else:
        freqh, freqw = freqs
    ## Compute gradients in Fourier domain
    gxU = 2 * np.pi * freqw * (-torch.imag(U) + 1j * torch.real(U))
    gxU = torch.fft.ifftshift(gxU, dim=(-2, -1))
    gxu = torch.real(torch.fft.ifft2(gxU))
    # gxu = crop(gxu, (h, w))
    gyU = 2 * np.pi * freqh * (-torch.imag(U) + 1j * torch.real(U))
    gyU = torch.fft.ifftshift(gyU, dim=(-2, -1))
    gyu = torch.real(torch.fft.ifft2(gyU))
    # gyu = crop(gyu, (h, w))
    return gxu, gyu


def crop(image, new_size):
    size = image.shape[-2:]
    if size[0] - new_size[0] > 0:
        image = image[..., :new_size[0], :]
    if size[1] - new_size[1] > 0:
        image = image[..., :new_size[1]]
    return image


def gaussian_filter(sigma, theta, shift=np.array([0.0, 0.0]), k_size=np.array([15, 15])):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1, lambda_2 = sigma
    theta = -theta

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1**2, lambda_2**2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position
    MU = k_size // 2 - shift
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calculate Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ))

    # Normalize the kernel and return
    if np.sum(raw_kernel) < 1e-2:
        kernel = np.zeros_like(raw_kernel)
        kernel[k_size[0]//2, k_size[1]//2] = 1
    else:
        kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


def dirac(dims):
    kernel = zeros(dims)
    hh = dims[0] // 2
    hw = dims[1] // 2
    kernel[hh, hw] = 1
    return kernel


def gaussian(images, sigma=1.0, theta=0.0):
    ## format Gaussian parameter for the gaussian_filter routine
    if isinstance(sigma, float) or isinstance(sigma, int):
        sigmas = ones(images.shape[0],2) * sigma
    elif isinstance(sigma, tuple) or isinstance(sigma, list):
        sigmas = ones(images.shape[0],2)
        sigmas[:,0] *= sigma[0]
        sigmas[:,1] *= sigma[1]
    else:
        sigmas = sigma
    if isinstance(theta, float) or isinstance(theta, int):
        thetas = ones(images.shape[0],1) * theta
    else:
        thetas = theta
    assert(theta.ndim-2)
    ## perform Gaussian filtering
    kernels = gaussian_filter(sigmas=sigmas, thetas=thetas)
    kernels = torch.to_tensor(kernels).unsqueeze(1).float().to(images.device)  # Nx1xHxW
    return conv2d(images, kernels)



def images_gradients(images, sigma=1.0):
    images_smoothed = fast_gaussian(images, sigma)
    gradients_x = torch.roll(images_smoothed, 1, dims=-2) - torch.roll(images_smoothed, -1, dims=-2)
    gradients_y = torch.roll(images_smoothed, 1, dims=-1) - torch.roll(images_smoothed, -1, dims=-1)
    return gradients_x, gradients_y



#####################################################################
####################### Fourier kernel ##############################
#####################################################################


### From here, taken from https://github.com/cszn/USRNet/blob/master/utils/utils_deblur.py
def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        # otf: NxCxHxWx2
        otf: NxCxHxW
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fft2(otf, dim=(-2, -1))
    return otf
