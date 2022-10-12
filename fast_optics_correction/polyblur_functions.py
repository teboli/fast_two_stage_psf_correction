import numpy as np
import torch
import torch.fft
import torch.nn.functional as F

from . import utils
from . import utils_fft
from . import filters
from . import edgetaper

from time import time

#########################################
############# Deblurring ################
#########################################


def mild_inverse_rank3(img, kernel, alpha=2, b=4, correlate=False, do_halo_removal=False, 
                       do_edgetaper=False, grad_img=None, method='direct'):
    """
    Deconvolution with approximate inverse filter parameterized by alpha and beta. (Deconvolution Alg.4, EdgeAwareFiltering is in Alg.1)
    :param img: (B,C,H,W) torch.tensor, the blurry image(s)
    :param kernel: (B,C,h,w) torch.tensor, the blur kernel(s)
    :param alpha: float, mid frequencies parameter
    :param b: float, high frequencies parameter
    :param correlate: bool, deconvolving with a correlation or not
    :param do_halo_removal: bool, using or not halo removal masking
    :param do_edgetaper bool, using or not edgetaping border preprocessing for deblurring
    :param method string, weither running the convolution in time or fourier domain
    :return torch.tensor of same size as img, the deblurred image(s)
    """
    if correlate:
        kernel = torch.rot90(kernel, k=2, dims=(-2, -1))
    ## Pad (with edgetaping optionally)
    img = utils.pad_with_kernel(img, kernel)
    if do_edgetaper:
        img = edgetaper.edgetaper(img, kernel, method=method)  # for better edge handling
    ## Deblurring
    imout = compute_polynomial(img, kernel, alpha, b, method=method)
    ## Crop
    imout = utils.crop_with_kernel(imout, kernel)
    ## Mask deblurring halos
    if do_halo_removal:
        img = utils.crop_with_kernel(img, kernel)
        imout = halo_masking(img, imout, grad_img)
    return torch.clamp(imout, 0.0, 1.0)


def compute_polynomial(img, kernel, alpha, b, method='fft'):
    if method == 'fft':
        return compute_polynomial_fft(img, kernel, alpha, b)
    elif method == 'direct':
        return compute_polynomial_direct(img, kernel, alpha, b)
    else:
        Exception('%s not implemented' % method)


def compute_polynomial_direct(img, kernel, alpha, b):
    """
    Implements in the time domain the polynomial deconvolution filter (Deconvolution Alg.4) 
    using the polynomial approximation of Eq. (27)
    :param img: (B,C,H,W) torch.tensor, the blurry image(s)
    :param kernel: (B,C,h,w) or (B,1,h,w) torch.tensor, the blur kernel(s)
    :param alpha: float, mid frequencies parameter for deblurring
    :param beta: float, high frequencies parameter for deblurring
    :return torch.tensor of same size as img, the deblurred image(s)
    """
    a3 = alpha/2 - b + 2
    a2 = 3 * b - alpha - 6
    a1 = 5 - 3 * b + alpha / 2
    imout = a3 * img
    imout = filters.convolve2d(imout, kernel) + a2 * img
    imout = filters.convolve2d(imout, kernel) + a1 * img
    return filters.convolve2d(imout, kernel) + b * img


def compute_polynomial_fft(img, kernel, alpha, b):
    """
    Implements in the fourier domain the polynomial deconvolution filter (Deconvolution Alg.4) 
    using the polynomial approximation of Eq. (27)
    :param Y: (B,C,H,W) torch.tensor, the blurry image(s)
    :param K: (B,C,h,w)  or (B,1,h,w) torch.tensor, the blur kernel(s)
    :param alpha: float, mid frequencies parameter for deblurring
    :param beta: float, high frequencies parameter for deblurring
    :return torch.tensor of same size as img, the deblurred image(s)
    """
    ## Go to Fourier domain
    h, w = img.shape[-2:]
    Y = torch.fft.fft2(img, dim=(-2, -1))
    K = filters.p2o(kernel, (h, w))  # from NxCxhxw to NxCxHxW
    a3 = alpha / 2 - b + 2
    a2 = 3 * b - alpha - 6
    a1 = 5 - 3 * b + alpha / 2
    X = a3 * Y
    X = K * X + a2 * Y
    X = K * X + a1 * Y
    X = K * X + b * Y
    ## Go back to temporal domain
    return torch.real(torch.fft.ifft2(X, dim=(-2, -1)))


# @torch.jit.script
def grad_prod_(grad_x, grad_y, gout_x, gout_y):
    return (- grad_x * gout_x) +  (- grad_y * grad_y)


# @torch.jit.script
def grad_square_(grad_x, grad_y):
    return grad_x * grad_x + grad_y * grad_y


# @torch.jit.script
def grad_div_and_clip_(M, nM):
    return torch.clamp(M / (nM + M), min=0)


# @torch.jit.script
def grad_convex_sum_(img, imout, z):
    # Equivalent to z * img + (1-z) * imout
    return imout + z * (img - imout)


def halo_masking(img, imout, grad_img=None):
    """
    Halo removal processing. Detects gradient inversions between input and deblurred image replaces them in the output (Alg.5)
    :param img: (B,C,H,W) torch.tensor, the blurry image(s)
    :param imout: (B,C,H,W) torch.tensor, the deblurred image(s)
    :return torch.tensor of same size as img, the halo corrected image(s)
    """
    if grad_img is None:
        grad_x, grad_y = filters.fourier_gradients(img)
    else:
        grad_x, grad_y = grad_img
    gout_x, gout_y = filters.fourier_gradients(imout)
    M = grad_prod_(grad_x, grad_y, gout_x, gout_y)
    nM = torch.sum(grad_square_(grad_x, grad_y), dim=(-2, -1), keepdim=True)
    z = grad_div_and_clip_(M, nM)
    return grad_convex_sum_(img, imout, z)


#########################################
########## Blur estimation ##############
#########################################


def blur_estimation(img, sigma_b, c, ker_size, q, thetas, interpolated_thetas, freqs=None):
    # flag saturated areas and edges
    start = time()
    mask = compute_mask(img)
    print('    mask:      %1.3f' % (time() - start))

    # normalized images
    start = time()
    img_normalized = normalize(img, q=q)
    print('    normaliz:  %1.3f' % (time() - start))

    # compute the image gradients
    start = time()
    gradients = compute_gradients(img_normalized, mask, freqs)
    print('    gradient:  %1.3f' % (time() - start))

    # compute the gradiennt magnitudes per orientation
    start = time()
    gradients_magnitude = compute_gradient_magnitudes(gradients, thetas)
    print('    magntiude: %1.3f' % (time() - start))

    # find the maximal direction amongst sampled orientations
    start = time()
    magnitude_normal, magnitude_ortho, theta = find_blur_direction(gradients, gradients_magnitude,
                                                                   thetas, interpolated_thetas)
    print('    direction: %1.3f' % (time() - start))

    # compute the Gaussian parameters
    start = time()
    sigma, rho = compute_gaussian_parameters(magnitude_normal, magnitude_ortho, c=c, sigma_b=sigma_b)
    print('    params:    %1.3f' % (time() - start))

    # create the blur kernel
    start = time()
    kernel = create_gaussian_filter(theta, sigma, rho, ksize=ker_size)
    print('    kernel:    %1.3f' % (time() - start))

    return kernel


def compute_mask(img, crop=11, threshold=0.97):
    ## First discards the saturated pixels
    mask = img > threshold
    ## Second discards the edges
    mask[..., :crop, :] = 0
    mask[..., -crop:, :] = 0
    mask[..., :, :crop] = 0
    mask[..., :, -crop:] = 0
    return mask


def normalize(img, q=0.0001):
    if q > 0:
        b, c, h, w = img.shape
        value_min = torch.quantile(img.reshape(b, c, -1), q=q, dim=-1, keepdim=True).unsqueeze(-1)  # (b,c,1,1)
        value_max = torch.quantile(img.reshape(b, c, -1), q=1-q, dim=-1, keepdims=True).unsqueeze(-1)  # (b,c,1,1)
        img = (img - value_min) / (value_max - value_min)
        return img.clamp(0.0, 1.0)
    else:
        value_min = torch.amin(img, dim=(-1,-2), keepdim=True)
        value_max = torch.amax(img, dim=(-1,-2), keepdim=True)
        return (img - value_min) / (value_max - value_min)


def compute_gradients(img, mask, freqs):
    gradient_x, gradient_y = filters.fourier_gradients(img, freqs)
    gradient_x[mask] = 0
    gradient_y[mask] = 0
    return gradient_x, gradient_y


def compute_gradient_magnitudes(gradients, angles):
    gradient_x, gradient_y = gradients  # (B,C,H,W)
    gradient_x_gray = gradient_x.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    gradient_y_gray = gradient_y.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
    angles = (angles / 180 * np.pi).view(1, -1, 1, 1, 1)
    # angles = torch.linspace(0, np.pi, n_angles + 1, device=gradient_x.device).view(1, -1, 1, 1, 1)  # (1,N,1,1,1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    gradient_magnitudes_angles = (cos * gradient_x_gray - sin * gradient_y_gray).abs()  # (B,N,1,H,W)
    gradient_magnitudes_angles = torch.amax(gradient_magnitudes_angles, dim=(-3, -2, -1))  # (B,N)
    return gradient_magnitudes_angles


# @torch.jit.script
def cubic_interpolator(x_new, x, y):
    """
    Fast implement of cubic interpolator based on Keys' algorithm
    """
    x_new = torch.abs(x_new[..., None] - x[..., None, :])
    mask1 = x_new < 1
    mask2 = torch.bitwise_and(1 <= x_new, x_new < 2)
    x_new = mask2 * (((-0.5 * x_new + 2.5) * x_new - 4) * x_new + 2) + \
            mask1 * ((1.5 * x_new - 2.5) * x_new * x_new + 1) 
    x_new /= torch.sum(x_new, dim=-1, keepdim=True) + 1e-5
    return (x_new * y.unsqueeze(1)).sum(dim=-1)


def find_blur_direction(gradients, gradient_magnitudes_angles, thetas, interpolated_thetas):
    gradient_x, gradient_y = gradients
    b = gradient_x.shape[0]
    n_interpolated_angles = interpolated_thetas.shape[-1]
    ## Find interpolated magnitudes at all thetas
    gradient_magnitudes_interpolated_angles = cubic_interpolator(interpolated_thetas / n_interpolated_angles,
                                                thetas / n_interpolated_angles, gradient_magnitudes_angles)  # (B,N)
    # print(gradient_magnitudes_interpolated_angles)
    ## Compute magnitude in theta
    i_min = torch.argmin(gradient_magnitudes_interpolated_angles, dim=-1, keepdim=True).long()
    thetas_normal = torch.take_along_dim(interpolated_thetas, i_min, dim=-1)
    # magnitudes_normal = torch.take_along_dim(gradient_magnitudes_interpolated_angles, i_min, dim=-1)
    gradient_color_magnitude_normal = gradient_x * torch.cos(thetas_normal.view(-1, 1, 1, 1) * np.pi / 180) - \
                                      gradient_y * torch.sin(thetas_normal.view(-1, 1, 1, 1) * np.pi / 180)
    magnitudes_normal = torch.abs(gradient_color_magnitude_normal)
    magnitudes_normal = torch.amax(magnitudes_normal.view(b, 3, -1), dim=-1)
    ## Compute magnitude in theta + 90
    thetas_ortho = (thetas_normal + 90.0) % 180  # angle in [0,pi)
    # i_ortho = (thetas_ortho / (180 / n_interpolated_angles)).long()
    # magnitudes_ortho = torch.take_along_dim(gradient_magnitudes_interpolated_angles, i_ortho, dim=-1)
    gradient_color_magnitude_ortho = gradient_x * torch.cos(thetas_ortho.view(-1, 1, 1, 1) * np.pi / 180) - \
                                     gradient_y * torch.sin(thetas_ortho.view(-1, 1, 1, 1) * np.pi / 180)
    magnitudes_ortho = torch.abs(gradient_color_magnitude_ortho)
    magnitudes_ortho = torch.amax(magnitudes_ortho.view(b, 3, -1), dim=-1)
    return magnitudes_normal, magnitudes_ortho, thetas_normal * np.pi / 180  # (B,3), (B,3), (B)


def compute_gaussian_parameters(magnitudes_normal, magnitudes_ortho, c, sigma_b):
    cc = c * c
    bb = sigma_b * sigma_b
    ## Compute sigma
    sigma = cc / (magnitudes_normal * magnitudes_normal + 1e-8) - bb
    sigma = torch.clamp(sigma, min=0.09, max=16.0)
    sigma = torch.sqrt(sigma)
    ## Compute rho
    rho = cc / (magnitudes_ortho * magnitudes_ortho + 1e-8) - bb
    rho = torch.clamp(rho, min=0.09, max=16.0)
    rho = torch.sqrt(rho)
    return sigma, rho


# @torch.jit.script
def compute_gaussian_filter_parameters(sigmas, rhos, thetas):
    B = len(sigmas)
    C = 3
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = sigmas
    lambda_2 = rhos
    thetas = -thetas  # minus sign because y-axis gradient is computed upside down

    # Set COV matrix using Lambdas and Theta
    c = torch.cos(thetas)
    s = torch.sin(thetas)
    cc = c*c
    ss = s*s
    sc = s*c
    inv_lambda_1 = 1.0 / (lambda_1 * lambda_1)
    inv_lambda_2 = 1.0 / (lambda_2 * lambda_2)
    inv_sigma00 = cc * inv_lambda_1 + ss * inv_lambda_2
    inv_sigma01 = sc * (inv_lambda_1 - inv_lambda_2)
    inv_sigma11 = cc * inv_lambda_2 + ss * inv_lambda_1
    return inv_sigma00, inv_sigma01, inv_sigma11


def create_gaussian_filter(thetas, sigmas, rhos, ksize):
    B = len(sigmas)
    C = 3

    # Create the inverse of the covariance matrix
    INV_SIGMA00, INV_SIGMA01, INV_SIGMA11 = compute_gaussian_filter_parameters(sigmas, rhos, thetas)
    INV_SIGMA = torch.stack([torch.stack([INV_SIGMA00, INV_SIGMA01], dim=-1),
                             torch.stack([INV_SIGMA01, INV_SIGMA11], dim=-1)], dim=-2)
    INV_SIGMA = INV_SIGMA.view(B, C, 1, 1, 2, 2)  # (B,C,1,1,2,2)

    # Create meshgrid for Gaussian
    t = torch.arange(ksize, device=sigmas.device) - ((ksize-1) // 2)
    X, Y = torch.meshgrid(t, t, indexing='xy')
    Z = torch.stack([X, Y], dim=-1).float()  # (k,k,2)

    # Calculate Gaussian for every pixel of the kernel
    COV = Z[..., 0] * (Z[...,0] * INV_SIGMA[...,0,0] + Z[...,1] * INV_SIGMA[...,0,1]) + \
          Z[..., 1] * (Z[...,0] * INV_SIGMA[...,1,0] + Z[...,1] * INV_SIGMA[...,1,1])
    kernels = torch.exp(-0.5 * COV)  # (B,C,k,k)
    return kernels / ( torch.sum(kernels, dim=(-1,-2), keepdim=True) + 1e-5)

