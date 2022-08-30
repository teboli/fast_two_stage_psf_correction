import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate

from . import filters
from .filters import gaussian_filter, fourier_gradients
from skimage import exposure, morphology, color
from scipy import interpolate, ndimage
import cv2

from .utils_float32 import *
from .utils_fft import psf2otf
from registration import max_phase_correlation_masked



######################################
############ Vanilla #################
######################################


def gaussian_blur_estimation(images, q=0.0001, n_angles=6, c=89.8, sigma_b=0.764, ksize=25, scale=1):
    # normalized image
    normalized_images = normalize(images, q=q)
    # compute the image gradients
    gradients = compute_gradients(normalized_images)
    # compute the gradient magntiudes per orientation
    gradient_magnitudes = compute_gradient_magnitudes(gradients, n_angles=n_angles)
    # find the maximal blur direction amongst sampled orientations
    directions = find_maximal_blur_direction(gradient_magnitudes, n_angles=n_angles)
    # finally compute the Gaussian parameters
    sigma_0, rho, theta = compute_gaussian_parameters(directions, c=c, sigma_b=sigma_b)
    # create the blur kernel
    kernel = create_gaussian_filter(sigma_0, rho, theta, k_size=ksize)
    # rescale kernel if needed, e.g., from raw image
    if scale > 1:
        kernel = F.interpolate(kernel, scale_factor=scale, mode='bilinear')
    return kernel, sigma_0, rho, theta


def normalize(images, q=0.0001):
    value_min = torch.quantile(images, q=q, dim=(1,2), keepdim=True)
    value_max = torch.quantile(images, q=1.0-q, dim=(1,2), keepdim=True)
    images_rescaled = (images - value_min) / (value_max - value_min)
    images_rescaled = torch.clamp(images_rescaled, 0.0, 1.0)
    return images_rescaled


def compute_gradients(images):
    gradients_x, gradients_y = fourier_gradients(images)
    return gradients_x, gradients_y


def compute_gradient_magnitudes(gradients, n_angles=6):
    gradient_x, gradient_y = gradients
    gradient_magnitudes = torch.zeros(gradients.shape[0], n_angles, dtype=gradients.dtype)
    for i in range(n_angles):
        angle = i * np.pi / n_angles
        gradient_at_angle = gradient_x * torch.cos(angle) - gradient_y * torch.sin(angle)
        gradient_magnitude_at_angle = torch.amax(torch.abs(gradient_at_angle), dim=(-2,-1))
        gradient_magnitudes[:,i] = gradient_magnitude_at_angle
    return gradient_magnitudes


def find_maximal_blur_direction(gradient_magnitudes, n_angles=6):
    # first build all sampled orientations
    all_thetas = array([i*180.0/n_angles for i in range(n_angles)])
    # get the maximal magnitude
    i_min = torch.argmin(gradient_magnitudes)
    theta_normal = all_thetas[i_min]
    magnitude_normal = gradient_magnitudes[i_min]
    # get orthogonal magnitude
    theta_ortho = (theta_normal + 90.0) % 180  # angle in [0,pi)
    i_ortho = int(theta_ortho // (180 / n_angles))
    magnitude_ortho = gradient_magnitudes[i_ortho]
    return magnitude_normal, magnitude_ortho, theta_normal * np.pi / 180


def compute_gaussian_parameters(directions, c=89.8, sigma_b=0.764):
    magnitude_normal, magnitude_ortho, theta = directions
    sigma_0 = torch.sqrt(torch.maximum(c**2/magnitude_normal**2 - sigma_b**2, 1e-8))
    sigma_1 = torch.sqrt(torch.maximum(c**2/magnitude_ortho**2 - sigma_b**2, 1e-8))
#     sigma_0 = c / magnitude_normal
#     sigma_1 = c / magnitude_ortho
    rho = sigma_1 / sigma_0
    return sigma_0, rho, theta


def create_gaussian_filter(sigma_0, rho, theta, ksize=25):
    sigma_1 = sigma_0 * rho
    kernel = torch.zeros(theta.shape[0], 1, ksize, ksize, dtype=theta.dtype)
    kernel[ksize//2, ksize//2] = 1
    ksize = np.array([ksize, ksize])
    kernel = gaussian_filter(sigma=(sigma_1, sigma_0), theta=theta, k_size=ksize)
    for n in range(sigma_0.shape[0]):
        kernel[n:n+1] = rotate(kernel[n:n+1], angle=theta*180/np.pi)
    kernel /= kernel.sum(dim=(-2,-1), keepdim=True)
    return kernel



##############################################
############ Vanilla (numpy) #################
##############################################


def gaussian_blur_estimation_np(img, q=0.0001, n_angles=6, c=89.8, sigma_b=0.764, ksize=25):
    # normalized image
    normalized_img = normalize_np(img, q=q)
    # compute the image gradients
    gradients = compute_gradients_np(normalized_img)
    # compute the gradient magntiudes per orientation
    gradient_magnitudes = compute_gradient_magnitudes_np(gradients, n_angles=n_angles)
    # find the maximal blur direction amongst sampled orientations
    directions = find_maximal_blur_direction_np(gradient_magnitudes, n_angles=n_angles)
    # finally compute the Gaussian parameters
    sigma_0, rho, theta = compute_gaussian_parameters_np(directions, c=c, sigma_b=sigma_b)
    # create the blur kernel
    kernel = create_gaussian_filter_np(sigma_0, rho, theta, ksize=ksize)
    return kernel, sigma_0, rho, theta


def normalize_np(img, q=0.0001):
    # value_min = np.quantile(img, q=q)
    # value_max = np.quantile(img, q=1.0-q)
    # return exposure.rescale_intensity(img, (value_min, value_max))
    # value_min = np.quantile(img, q=q, axis=(0, 1), keepdims=True)
    # value_max = np.quantile(img, q=1.0-q, axis=(0, 1), keepdims=True)
    img_normalized = np.zeros_like(img)
    for i in range(3):
        value_min = np.quantile(img[..., i], q=q)
        value_max = np.quantile(img[..., i], q=1.0-q)
        img_normalized[..., i] = exposure.rescale_intensity(img[..., i], (value_min, value_max))
    # print(img_normalized.max(), img_normalized.min())
    return img_normalized


def compute_gradients_np(img):
    gradient_x, gradient_y = fourier_gradients(img)
    return gradient_x, gradient_y


def compute_gradient_magnitudes_np(gradients, n_angles=6):
    gradient_x, gradient_y = gradients
    gradient_magnitudes = zeros(n_angles)
    for i in range(n_angles):
        angle = i * np.pi / n_angles
        gradient_at_angle = gradient_x * np.cos(angle) - gradient_y * np.sin(angle)
        gradient_magnitude_at_angle = np.amax(np.abs(gradient_at_angle))
        gradient_magnitudes[i] = gradient_magnitude_at_angle
    return gradient_magnitudes


def find_maximal_blur_direction_np(gradient_magnitudes, n_angles=6):
    # first build all sampled orientations
    all_thetas = array([i*180.0/n_angles for i in range(n_angles)])
    # get the maximal magnitude
    i_min = np.argmin(gradient_magnitudes)
    theta_normal = all_thetas[i_min]
    magnitude_normal = gradient_magnitudes[i_min]
    # get orthogonal magnitude
    theta_ortho = (theta_normal + 90.0) % 180  # angle in [0,pi)
    i_ortho = int(theta_ortho // (180 / n_angles))
    magnitude_ortho = gradient_magnitudes[i_ortho]
    return magnitude_normal, magnitude_ortho, theta_normal * np.pi / 180


def compute_gaussian_parameters_np(directions, c=89.8, sigma_b=0.764):
    magnitude_normal, magnitude_ortho, theta = directions
    sigma_0 = np.sqrt(np.maximum(c**2/magnitude_normal**2 - sigma_b**2, 1e-8))
    sigma_1 = np.sqrt(np.maximum(c**2/magnitude_ortho**2 - sigma_b**2, 1e-8))
#     sigma_0 = c / magnitude_normal
#     sigma_1 = c / magnitude_ortho
    rho = sigma_1 / sigma_0
    return sigma_0, rho, theta


def create_gaussian_filter_np(sigma_0, rho, theta, ksize=25):
    sigma_1 = sigma_0 * rho
    sigma = (sigma_0, sigma_1)
    ksize = np.array([ksize, ksize])
    # kernel = zeros((ksize,ksize))
    # kernel[ksize//2, ksize//2] = 1
    # kernel = filters.gaussian(kernel, sigma=(sigma_1, sigma_0), mode='nearest')
    # kernel = transform.rotate(kernel, angle=theta*180/np.pi)
    # kernel /= kernel.sum()
    kernel = gaussian_filter(sigma=sigma, theta=theta, k_size=ksize)
    return kernel




##############################################
############ Colored (numpy) #################
##############################################



def rgb_gaussian_blur_estimation(img, theta_normal=None, q=0.0001, n_angles=6, n_interpolated_angles=30, c=89.8, sigma_b=0.764, ksize=25):
    if type(c) == float:
        c = (c, c)
    if type(sigma_b) == float:
        sigma_b = (sigma_b, sigma_b)
    # normalized image
    normalized_img = normalize_np(img, q=q)
    # Go to grayscale
    # normalized_img = color.rgb2gray(normalized_img)
    # normalized_img = np.stack([normalized_img, normalized_img, normalized_img], axis=-1)
    # compute the image gradients
    gradients = fourier_gradients(normalized_img)
    # compute the gradient magnitudes per orientation
    gradient_magnitudes_gray = compute_gray_gradient_magnitudes(gradients, n_angles=n_angles)  # RGB update
    # find the maximal blur direction amongst interpolated orientations
    directions = find_maximal_rgb_blur_parameters(normalized_img, gradients, gradient_magnitudes_gray,
                                                  theta_normal=theta_normal, n_angles=n_angles,
                                                  n_interpolated_angles=n_interpolated_angles)
    # compute the shifts
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    # gr = (g - g.mean()) / g.std() * r.std() + r.mean()
    # gb = (g - g.mean()) / g.std() * b.std() + b.mean()
    # shift_r = phase_correlate(r, g)
    # shift_g = np.zeros(2)
    # shift_b = phase_correlate(b, g)
    # finally compute the Gaussian parameters
    sigma_0, rho, theta, shift = compute_rgb_gaussian_parameters(directions, c=c, sigma_b=sigma_b)
    # print(shift)
    # shift = [shift_r, shift_g, shift_b]
    # print(shift)
    # create the blur kernel
    kernel = create_rgb_gaussian_filter(sigma_0, rho, theta, shift, ksize=ksize)
    return kernel, sigma_0, rho, theta, shift


def compute_gray_gradient_magnitudes(gradients, n_angles=6):
    gradient_x, gradient_y = gradients  # (h,w,c)
    gradient_x_gray = gradient_x.mean(-1)
    gradient_y_gray = gradient_y.mean(-1)
    h, w = gradient_x.shape[:2]  # (h,w)
    gradient_magnitudes_gray = zeros(n_angles+1)
    angles = linspace(0, np.pi, n_angles+1)
    for i in range(n_angles+1):
        angle = angles[i]
        gradient_gray_at_angle = gradient_x_gray * np.cos(angle) - gradient_y_gray * np.sin(angle)
        gradient_magnitude_gray_at_angle = np.amax(np.abs(gradient_gray_at_angle))
        gradient_magnitudes_gray[i] = gradient_magnitude_gray_at_angle
    return gradient_magnitudes_gray


# def find_maximal_rgb_blur_parameters(img, gradients, gradient_magnitudes_gray, theta_normal=None, n_angles=6, n_interpolated_angles=30, winsize=5):
#     ## first build all sampled orientations, with interpolated ones
#     all_thetas = linspace(0, np.pi, n_angles+1)
#     all_interpolated_thetas = linspace(0, np.pi, n_interpolated_angles+1)
#     ## interpolate grayscale gradients at all interpolated thetas
#     f_gray = interpolate.interp1d(all_thetas, gradient_magnitudes_gray, kind='cubic')
#     interpolated_gradient_magnitudes_gray = f_gray(all_interpolated_thetas)
#     ## get theta normal and orthogonal from grayscale gradients
#     i_min = np.argmin(interpolated_gradient_magnitudes_gray)
#     if theta_normal is None:
#         theta_normal = all_interpolated_thetas[i_min]  # angle in [0,pi]
#     theta_ortho = (theta_normal * 180 / np.pi + 90.0) % 180  # angle in [0,180]
#     theta_ortho *= np.pi / 180  # angle in [0,pi]
#     ## Estimate blur size for each color channel at normal direction from RGB gradients
#     magnitude_normal = zeros(3)
#     ### Start with estimating the green normal component as reference
#     gradient_x, gradient_y = gradients
#     gradient_at_normal_angle = gradient_x * np.cos(theta_normal) - gradient_y * np.sin(theta_normal)
#     ### Discard saturated pixel to not blow the algorithm up and predict (ig,jg)
#     saturation_mask = img > 0.99
#     saturation_mask_dilated = morphology.binary_dilation(saturation_mask, selem=morphology.disk(winsize)[:,:,None])
#     magnitude_at_normal_angle = np.abs(gradient_at_normal_angle)
#     masked_magnitude_at_normal_angle = array(np.abs(gradient_at_normal_angle))
#     masked_magnitude_at_normal_angle[saturation_mask_dilated] = 0
#     magnitude_normal[1] = np.amax(masked_magnitude_at_normal_angle[..., 1])   # select the maximal gradient value at theta_normal angle
#     # ### Start with estimating the green normal component as reference
#     # gradient_x, gradient_y = gradients
#     # gradient_magnitudes = []
#     # for angle in [0, 30, 60, 90, 120, 150, 180]:
#     #     tt = angle * np.pi / 180
#     #     gradient_magnitudes.append(np.abs(gradient_x * np.cos(tt) - gradient_y * np.sin(tt)))
#     #     gradient_magnitudes[-1][saturation_mask_dilated] = 0
#     #     gradient_magnitudes[-1] = np.amax(gradient_magnitudes[-1])
#     # gradient_magnitudes = np.stack(gradient_magnitudes)
#     # ff_red = interpolate.interp1d(all_thetas, gradient_magnitudes[..., 0], kind='cubic')
#     # ff_green = interpolate.interp1d(all_thetas, gradient_magnitudes[..., 1], kind='cubic')
#     # ff_blue = interpolate.interp1d(all_thetas, gradient_magnitudes[..., 2], kind='cubic')
#     # i_normal = int(theta_normal * 180 / np.pi)
#     # magnitude_normal[1] = ff_green(all_interpolated_thetas)[i_normal]
#     ### Get the coordinates of the green value and of the line passing through the center and the argmax of the green channel
#     h, w = gradient_x.shape[:2]
#     ig, jg = np.unravel_index(np.argmax(masked_magnitude_at_normal_angle[..., 1]), (h, w))  # Get first the location
#     ### /!\ radial model of the chromatic aberrations:
#     ### Find along a line oriented by theta_normal and passing through the green value
#     ### the relative sub-pixel coordinates of red and blue maxima in a region 
#     ### next to the green value. Do it with the Canny edge detector-style algorithm
#     linesize = winsize
#     i_on_line = -np.sin(theta_normal) * linspace(-(winsize//2), winsize//2, linesize) + ig  # sub-pixel segment passing by (ig, jg)
#     j_on_line = np.cos(theta_normal) * linspace(-(winsize//2), winsize//2, linesize) + jg
#     ### Find the closest 4 points on the regular grid and within the image boundaries

#     def clip(index, limit):
#         return np.clip(index, 0, limit-1).astype(np.int64)

#     i_top_left = clip(np.floor(i_on_line), h-1)
#     j_top_left = clip(np.floor(j_on_line), w-1)
#     i_top_right = clip(np.floor(i_on_line), h-1)
#     j_top_right = clip(np.ceil(j_on_line), w-1)
#     i_bottom_left = clip(np.ceil(i_on_line), h-1)
#     j_bottom_left = clip(np.floor(j_on_line), w-1)
#     i_bottom_right = clip(np.ceil(i_on_line), h-1)
#     j_bottom_right = clip(np.ceil(j_on_line), w-1)
#     ### Evaluate the magnitudes on the regular grid
#     magnitude_top_left = np.abs(magnitude_at_normal_angle)[i_top_left, j_top_left]
#     magnitude_top_right = np.abs(magnitude_at_normal_angle)[i_top_right, j_top_right]
#     magnitude_bottom_left = np.abs(magnitude_at_normal_angle)[i_bottom_left, j_bottom_left]
#     magnitude_bottom_right = np.abs(magnitude_at_normal_angle)[i_bottom_right, j_bottom_right]
#     ### Bilinear interpolation of the magnitudes on the segment
#     w_top_left = ((j_bottom_right - j_on_line) * (i_bottom_right - i_on_line))[:, None] + 1e-8
#     w_top_right = ((j_on_line - j_bottom_left) * (i_bottom_left - i_on_line))[:, None] + 1e-8
#     w_bottom_left = ((j_top_right - j_on_line) * (i_on_line - i_top_right))[:, None] + 1e-8
#     w_bottom_right = ((j_on_line - j_top_left) * (i_on_line - i_top_left))[:, None] + 1e-8
#     interpolated_magnitude = w_top_left * magnitude_top_left + w_top_right * magnitude_top_right + w_bottom_left * magnitude_bottom_left + w_bottom_right * magnitude_bottom_right
#     interpolated_magnitude /= w_top_left + w_top_right + w_bottom_left + w_bottom_right
#     ### Find maxima for the red and green colors
#     n_inter_samples = 5 * linesize   # Compute number of sub-pixel points
#     i_on_line_interpolated = linspace(i_on_line[0], i_on_line[-1], n_inter_samples)
#     j_on_line_interpolated = linspace(j_on_line[0], j_on_line[-1], n_inter_samples)
#     X = arange(linesize)
#     Z = linspace(0, linesize-1, n_inter_samples)
#     f_red = interpolate.interp1d(X, interpolated_magnitude[..., 0], kind='cubic')   # interpolate the magnitude values
#     f_green = interpolate.interp1d(X, interpolated_magnitude[..., 1], kind='cubic')
#     f_blue = interpolate.interp1d(X, interpolated_magnitude[..., 2], kind='cubic')
#     magnitude_normal[0] = np.amax(f_red(Z))  # store the red maximum
#     magnitude_normal[2] = np.amax(f_blue(Z))  # store the blue maximum
#     ### Compute the translations of the maxima compared to the green channel along the normal direction
#     idx_red_argmax = np.argmax(f_red(Z))
#     ir, jr = i_on_line_interpolated[idx_red_argmax], j_on_line_interpolated[idx_red_argmax]
#     idx_green_argmax = np.argmax(f_green(Z))
#     igg, jgg = i_on_line_interpolated[idx_green_argmax], j_on_line_interpolated[idx_green_argmax]
#     idx_blue_argmax = np.argmax(f_blue(Z))
#     ib, jb = i_on_line_interpolated[idx_blue_argmax], j_on_line_interpolated[idx_blue_argmax]
#     # print("ig: %1.2f | igg: %1.2f | jg: %1.2f | jgg: %1.2f" % ( ig, igg, jg, jgg ))
#     print("ir: %1.2f | jr: %1.2f" % (ir, jr))
#     print("ig: %1.2f | jg: %1.2f" % (igg, jgg))
#     print("ib: %1.2f | jb: %1.2f" % (ib, jb))
#     translation_red = (ir - igg, jr - jgg)  # before
#     translation_blue = (ib - igg, jb - jgg)  # before
#     # translation_red = (igg - ir, jgg - jr)
#     # translation_blue = (igg - ib, jgg - jb)
#     ## Estimate the blur size for each color at the orthogonal direction
#     gradient_at_ortho_angle = gradient_x * np.cos(theta_ortho) - gradient_y * np.sin(theta_ortho)
#     magnitude_at_ortho_angle = np.abs(gradient_at_ortho_angle)
#     magnitude_ortho = np.amax(magnitude_at_ortho_angle, axis=(0, 1))  # (3,)
#     return magnitude_normal, magnitude_ortho, theta_normal, translation_red, translation_blue


def mask_gradients(gradients, img, crop=11, disksize=3):
    masked_gradients = array(gradients)
    saturation_mask = img > 0.99
    saturation_mask_dilated = morphology.binary_dilation(saturation_mask, selem=morphology.disk(disksize)[: , :, None])
    masked_gradients[saturation_mask_dilated] = 0
    # black_mask = gradients.mean(axis=-1) == 0
    # black_mask_dilated = morphology.binary_dilation(black_mask, selem=morphology.disk(disksize))
    # masked_gradients[black_mask_dilated, 0] = 0
    # masked_gradients[black_mask_dilated, 1] = 0
    # masked_gradients[black_mask_dilated, 2] = 0
    masked_gradients[:crop, :] = 0
    masked_gradients[-crop:, :] = 0
    masked_gradients[:, :crop] = 0
    masked_gradients[:, -crop:] = 0
    return masked_gradients


def phase_correlate(reference, moving, m=0.9):
    # shift_xy = cv2.phaseCorrelate(reference, moving)[0]
    # print(shift_xy)
    # return np.array(shift_xy)
    # edges_reference = cv2.Canny((reference * 255).astype(np.uint8), threshold1=30, threshold2=100)
    # edges_reference = edges_reference.astype(np.float32) / 255
    # edges_moving = cv2.Canny((moving * 255).astype(np.uint8), threshold1=30, threshold2=100)
    # edges_moving = edges_moving.astype(np.float32) / 255
    # edges_moving = (edges_moving - edges_moving.mean()) / edges_moving.std()
    # edges_moving = edges_moving * edges_reference.std() + edges_reference.mean()
    # shift_xy = cv2.phaseCorrelate(edges_reference, edges_moving)[0]
    # return np.array(shift_xy)
    shift_xy = max_phase_correlation_masked(reference, moving, m)
    return -np.array(shift_xy)


def subpixelic_translation_fft(image, shift_ij):
    H, W = image.shape[:2]
    if len(image.shape) == 2:
        image = image[..., None]
    I = fft2(image, axes=(0, 1))
    I = np.fft.fftshift(I, axes=(0, 1))

    u = (arange(H) - H//2) / H
    v = (arange(W) - W//2) / W
    u = u[:, None, None]
    v = v[None, :, None]

    D_ij = np.exp(-2 * np.pi * 1j * ((shift_ij[0]*u) + (shift_ij[1]*v)))
    I_ij = I * D_ij
    I_ij = np.fft.ifftshift(I_ij, axes=(0, 1))
    img_ij = ifft2(I_ij, axes=(0 ,1))
    img_ij = np.real(img_ij)
    img_ij = np.squeeze(img_ij, -1)
    return img_ij


def find_maximal_rgb_blur_parameters(img, gradients, gradient_magnitudes_gray, theta_normal=None, n_angles=6, n_interpolated_angles=30, winsize=11):
    ## first build all sampled orientations, with interpolated ones
    all_thetas = linspace(0, np.pi, n_angles+1)
    all_interpolated_thetas = linspace(0, np.pi, n_interpolated_angles+1)
    ## interpolate grayscale gradients at all interpolated thetas
    f_gray = interpolate.interp1d(all_thetas, gradient_magnitudes_gray, kind='cubic')
    interpolated_gradient_magnitudes_gray = f_gray(all_interpolated_thetas)
    ## get theta normal and orthogonal from grayscale gradients
    i_min = np.argmin(interpolated_gradient_magnitudes_gray)
    if theta_normal is None:
        theta_normal = all_interpolated_thetas[i_min]  # angle in [0,pi]
    theta_ortho = (theta_normal * 180 / np.pi + 90.0) % 180  # angle in [0,180]
    i_ortho = int(theta_ortho // (180 / (n_interpolated_angles + 1)))
    theta_ortho *= np.pi / 180  # angle in [0,pi]
    gradient_x, gradient_y = gradients
    ### Compute the normal components
    gradient_at_normal_angle = gradient_x * np.cos(theta_normal) - gradient_y * np.sin(theta_normal)


    _, idx_normal = compute_rgb_max_and_argmax(np.abs(gradient_at_normal_angle))
    # print(img.shape, idx_normal)
    # gradient_at_normal_angle = normalize_color_channels(gradient_at_normal_angle, img, idx_normal)
    magnitude_at_normal_angle = np.abs(gradient_at_normal_angle)
    masked_magnitude_at_normal_angle = mask_gradients(magnitude_at_normal_angle, img, crop=50)
    magnitude_normal = np.amax(masked_magnitude_at_normal_angle, axis=(0, 1))  # (3,)
    magnitude_normal, idx_normal = compute_rgb_max_and_argmax(masked_magnitude_at_normal_angle)


    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(img)
    # plt.plot(idx_normal[0][1], idx_normal[0][0], 'or', markersize=5)
    # plt.plot(idx_normal[1][1], idx_normal[1][0], 'og', markersize=5)
    # plt.plot(idx_normal[2][1], idx_normal[2][0], 'ob', markersize=5)
    # plt.show()

    # magnitude_normal = normalize_color_channels(magnitude_normal, img, idx_normal)
    ## Estimate the blur size for each color at the orthogonal direction
    gradient_at_ortho_angle = gradient_x * np.cos(theta_ortho) - gradient_y * np.sin(theta_ortho)


    _, idx_ortho = compute_rgb_max_and_argmax(np.abs(gradient_at_ortho_angle))
    # gradient_at_ortho_angle = normalize_color_channels(gradient_at_ortho_angle, img, idx_ortho)


    magnitude_at_ortho_angle = np.abs(gradient_at_ortho_angle)
    masked_magnitude_at_ortho_angle = mask_gradients(magnitude_at_ortho_angle, img, crop=50)
    # magnitude_ortho = np.amax(masked_magnitude_at_ortho_angle, axis=(0, 1))  # (3,)
    magnitude_ortho, idx_ortho = compute_rgb_max_and_argmax(masked_magnitude_at_ortho_angle)
    # magnitude_ortho = normalize_color_channels(magnitude_ortho, img, idx_ortho)
    ### Compute the per channel translation from normal angle
    crop_img = masked_magnitude_at_normal_angle
    # crop_img = img
    # translation_red = phase_correlate(crop_img[..., 1], crop_img[..., 0])
    translation_red = np.zeros(2)
    translation_green = np.zeros(2)
    # translation_blue = phase_correlate(crop_img[..., 1], crop_img[..., 2])
    translation_blue = np.zeros(2)
    translation = np.stack([translation_red, translation_green, translation_blue])
    return magnitude_normal, magnitude_ortho, theta_normal, translation


def compute_rgb_gaussian_parameters(directions, c, sigma_b, thresh=4.0):
    magnitude_normal, magnitude_ortho, theta, shift = directions
    sigma_0 = c[0]**2/(magnitude_normal**2 + 1e-8) - sigma_b[0]**2
    sigma_0[np.bitwise_or(sigma_0 < 0, sigma_0 > 100)] = 0.09
    sigma_0 = np.sqrt(sigma_0)
    if min(magnitude_normal) < 0.10:
        sigma_0 = 0.3 * np.ones(3)
    # sigma_0 = np.clip(sigma_0, 0.3, 4.0)
    sigma_1 = c[1] ** 2 / (magnitude_ortho**2 + 1e-8) - sigma_b[1] ** 2
    sigma_1[np.bitwise_or(sigma_1 < 0, sigma_1 > 100)] = 0.09
    sigma_1 = np.sqrt(sigma_1)
    if min(magnitude_ortho) < 0.10:
        sigma_1 = 0.3 * np.ones(3)
    # if sigma_1 > 4.0:
    #     sigma_1 = 0.3
    # sigma_1 = np.clip(sigma_1, 0.3, 4.0)

    # Compute rho
    rho = sigma_1 / sigma_0
    return sigma_0, rho, theta, shift


def compute_rgb_max_and_argmax(img):
    h, w = img.shape[:2]
    img_flat = img.reshape(-1, 3)
    ## Compute indexes
    idx_r = np.argmax(img_flat[..., 0], axis=0)
    idx_r = np.unravel_index(idx_r, (h, w))
    idx_g = np.argmax(img_flat[..., 1], axis=0)
    idx_g = np.unravel_index(idx_g, (h, w))
    idx_b = np.argmax(img_flat[..., 2], axis=0)
    idx_b = np.unravel_index(idx_b, (h, w))
    argmax = (idx_r, idx_g, idx_b)
    ## Compute max values
    max_r = img[idx_r[0], idx_r[1], 0]
    max_g = img[idx_g[0], idx_g[1], 1]
    max_b = img[idx_b[0], idx_b[1], 2]
    max = np.array([max_r, max_g, max_b])
    return max, argmax


# def normalize_color_channels(magnitudes, img, argmax):
#     print('R: mean: %2.4f | std: %2.4f' % (img[..., 0].mean(), img[..., 0].std()))
#     img_max_r = img[argmax[0][0], argmax[0][1], 0]
#     magnitudes[0] /= img_max_r
#     print('G: mean: %2.4f | std: %2.4f' % (img[..., 1].mean(), img[..., 1].std()))
#     img_max_g = img[argmax[1][0], argmax[1][1], 1]
#     magnitudes[1] /= img_max_g
#     print('B: mean: %2.4f | std: %2.4f' % (img[..., 2].mean(), img[..., 2].std()))
#     img_max_b = img[argmax[2][0], argmax[2][1], 2]
#     magnitudes[2] /= img_max_b
#     magnitudes *= img_max_g
#     # magnitudes *= min((img_max_r, img_max_g, img_max_b))
#     # magnitudes *= np.mean((img_max_r, img_max_g, img_max_b))
#     return magnitudes


def normalize_color_channels(magnitudes, img, argmax):
    # mean = img.mean(axis=(0, 1))
    # std = img.std(axis=(0, 1))
    mean = magnitudes.mean(axis=(0, 1))
    std = magnitudes.std(axis=(0, 1))
    # mean = np.mean(img, axis=(0, 1))
    # std = np.std(img, axis=(0, 1))
    g = magnitudes
    gg = np.array(magnitudes)
    # gg /= std
    # gg *= np.mean(std)
    # gg -= mean
    gg /= std
    gg *= std[1]
    # gg += mean[1]
    mag = np.abs(magnitudes)
    mmag = np.abs(gg)

    print('#### RED ####')
    print('grad: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (g[..., 0].mean(), g[..., 0].std(), g[..., 0].max()))
    print('ngra: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (gg[..., 0].mean(), gg[..., 0].std(), gg[..., 0].max()))
    print('imag: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (img[..., 0].mean(), img[..., 0].std(), img[..., 0].max()))
    print('magn: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (mag[..., 0].mean(), mag[..., 0].std(), mag[..., 0].max()))
    print('nmag: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (mmag[..., 0].mean(), mmag[..., 0].std(), mmag[..., 0].max()))

    print('#### GREEN ####')
    print('grad: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (g[..., 1].mean(), g[..., 1].std(), g[..., 1].max()))
    print('ngra: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (gg[..., 1].mean(), gg[..., 1].std(), gg[..., 1].max()))
    print('imag: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (img[..., 1].mean(), img[..., 1].std(), img[..., 1].max()))
    print('magn: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (mag[..., 1].mean(), mag[..., 1].std(), mag[..., 1].max()))
    print('nmag: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (mmag[..., 1].mean(), mmag[..., 1].std(), mmag[..., 1].max()))

    print('#### BLUE ####')
    print('grad: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (g[..., 2].mean(), g[..., 2].std(), g[..., 2].max()))
    print('ngra: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (gg[..., 2].mean(), gg[..., 2].std(), gg[..., 2].max()))
    print('imag: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (img[..., 2].mean(), img[..., 2].std(), img[..., 2].max()))
    print('magn: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (mag[..., 2].mean(), mag[..., 2].std(), mag[..., 2].max()))
    print('nmag: mean: %2.4f | std: %2.4f | max_im: %1.2f' % (mmag[..., 2].mean(), mmag[..., 2].std(), mmag[..., 2].max()))
    print()



    # magnitudes /= mean
    # gg /= std
    # gg *= std[1]
    # magnitudes *= mean[1]
    # print('- R: mean: %2.4f | std: %2.4f' % (img[..., 0].mean(), img[..., 0].std()))
    # print('- G: mean: %2.4f | std: %2.4f' % (img[..., 1].mean(), img[..., 1].std()))
    # print('- B: mean: %2.4f | std: %2.4f' % (img[..., 2].mean(), img[..., 2].std()))
    return magnitudes


def subpxelic_translation(img, polar=None, cart=None):
    ## The translation is either given in polar or Cartesian coordinates.
    ## Subpixelic accuracy is done with bilinear interpolation of the peak.
    assert(polar is not None or cart is not None)
    if polar is not None:
        t, theta = polar
        ti = -np.sin(theta) * t
        tj = np.cos(theta) * t
    else:
        ti, tj = cart
    t_top_left = (np.floor(ti), np.floor(tj))
    t_top_right = (np.floor(ti), np.ceil(tj))
    t_bottom_left = (np.ceil(ti), np.floor(tj))
    t_bottom_right = (np.ceil(ti), np.ceil(tj))
    w_top_left = ((t_bottom_right[1] - tj) * (t_bottom_right[0] - ti)) + 1e-8
    w_top_right = ((tj - t_bottom_left[1]) * (t_bottom_left[0] - ti)) + 1e-8
    w_bottom_left = ((t_top_right[1] - tj) * (ti - t_top_right[0])) + 1e-8
    w_bottom_right = ((tj - t_top_left[1]) * (ti - t_top_left[0])) + 1e-8
    k = zeros(img.shape)
    hk = img.shape[0]//2
    k[int(hk + t_top_left[0]), int(hk + t_top_left[1])] = w_top_left
    k[int(hk + t_top_right[0]), int(hk + t_top_right[1])] = w_top_right
    k[int(hk + t_bottom_left[0]), int(hk + t_bottom_left[1])] = w_bottom_left
    k[int(hk + t_bottom_right[0]), int(hk + t_bottom_right[1])] = w_bottom_right
    k /= k.sum() + 1e-8


    K = psf2otf(k, img.shape[:2])
    I = fft2(img, axes=(0, 1))
    J = np.conj(K) * I
    out = np.real(ifft2(J))


    # out = ndimage.correlate(img, k)
    return out / (out.sum() + 1e-8)


def create_rgb_gaussian_filter(sigma_0, rho, theta, shift, ksize=25):
    sigma_1 = sigma_0 * rho
    sigma = np.stack((sigma_0, sigma_1), axis=-1)
    kernel = zeros((ksize, ksize, 3))
    k_size = np.array([ksize, ksize])
    # kernel[..., 0] = gaussian_filter(sigma=sigma[0], theta=theta, k_size=k_size, shift=shift[0]*0)
    # kernel[..., 1] = gaussian_filter(sigma=sigma[1], theta=theta, k_size=k_size, shift=shift[1]*0)
    # kernel[..., 2] = gaussian_filter(sigma=sigma[2], theta=theta, k_size=k_size, shift=shift[2]*0)
    kernel[..., 0] = gaussian_filter(sigma=sigma[0], theta=theta, k_size=k_size, shift=shift[0])
    kernel[..., 1] = gaussian_filter(sigma=sigma[1], theta=theta, k_size=k_size, shift=shift[1])
    kernel[..., 2] = gaussian_filter(sigma=sigma[2], theta=theta, k_size=k_size, shift=shift[2])
    return kernel



##############################################
############ Colored (pytorch) #################
##############################################


class GaussianBlurEstimator(nn.Module):
    def __init__(self, n_angles, n_interpolated_angles, c, sigma_b, k_size):
        super(GaussianBlurEstimator, self).__init__()
        self.n_angles = n_angles
        self.n_interpolated_angles = n_interpolated_angles
        self.c = c
        self.sigma_b = sigma_b
        self.k_size = k_size

    def forward(self, images):
        images_norm = self._normalize(images)
        gradients = filters.fourier_gradients(images_norm)
        gradient_magnitudes_angles = self._compute_magnitudes_at_angles(gradients)
        thetas, magnitudes_normal, magnitudes_ortho = self._find_direction(images_norm, gradients,
                                                                           gradient_magnitudes_angles)
        sigmas, rhos = self._find_variances(magnitudes_normal, magnitudes_ortho)
        kernels = self._create_gaussian_filter(thetas, sigmas, rhos)
        return kernels, (thetas, sigmas, rhos)

    def _normalize(self, images):
        images = (images - images.min()) / (images.max() - images.min())
        return images.clamp(0.0, 1.0)

    def _compute_magnitudes_at_angles(self, gradients):
        n_angles = self.n_angles
        gradient_x, gradient_y = gradients  # (B,C,H,W)
        gradient_x_gray = gradient_x.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
        gradient_y_gray = gradient_y.mean(1, keepdim=True).unsqueeze(1)  # (B,1,1,H,W)
        angles = torch.linspace(0, np.pi, n_angles + 1, device=gradient_x.device).view(1, -1, 1, 1, 1)  # (1,N,1,1,1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        gradient_magnitudes_angles = (cos * gradient_x_gray - sin * gradient_y_gray).abs()  # (B,N,1,H,W)
        gradient_magnitudes_angles = torch.amax(gradient_magnitudes_angles, dim=(-3, -2, -1))  # (B,N)
        return gradient_magnitudes_angles

    def _find_direction(self, images, gradients, gradient_magnitudes_angles):
        gradients_x, gradients_y = gradients  # ((B,C,H,W), (B,C,H,W))
        ## Find thetas
        gradient_magnitudes_angles = gradient_magnitudes_angles.unsqueeze(1)  # (B,1,N)
        gradient_magnitudes_interpolated_angles = F.interpolate(gradient_magnitudes_angles,
                                                                size=self.n_interpolated_angles,
                                                                mode='linear', align_corners=True).squeeze(1)  # (B,30)
        thetas = gradient_magnitudes_interpolated_angles.argmin(dim=1) * 180 / self.n_interpolated_angles  # (B)
        thetas = thetas.unsqueeze(-1)  # (B,1)
        ## Compute magnitude in theta
        cos = torch.cos(thetas).view(-1, 1, 1, 1)  # (B,1,1,1)
        sin = torch.sin(thetas).view(-1, 1, 1, 1)  # (B,1,1,1)
        magnitudes_normal = torch.amax((cos * gradients_x - sin * gradients_y).abs(), dim=(-2, -1))  # (B,C)
        ## Compute magnitude in theta+90
        cos = torch.cos(thetas + np.pi//2).view(-1, 1, 1, 1)  # (B,1,1,1)
        sin = torch.sin(thetas + np.pi//2).view(-1, 1, 1, 1)  # (B,1,1,1)
        magnitudes_ortho = torch.amax((cos * gradients_x - sin * gradients_y).abs(), dim=(-2, -1))  # (B,C)
        return thetas, magnitudes_normal, magnitudes_ortho

    def _find_variances(self, magnitudes_normal, magnitudes_ortho):
        a = self.c**2
        b = self.sigma_b**2
        ## Compute sigma
        sigma = a / (magnitudes_normal ** 2 + 1e-8) - b
        sigma[torch.bitwise_or(sigma < 0, sigma > 100)] = 0.09
        sigma = torch.sqrt(sigma)
        ## Compute rho
        rho = a / (magnitudes_ortho ** 2 + 1e-8) - b
        rho[torch.bitwise_or(rho < 0, rho > 100)] = 0.09
        rho = torch.sqrt(rho)
        return sigma, rho

    def _create_gaussian_filter(self, thetas, sigmas, rhos):
        k_size = self.k_size
        B, C = sigmas.shape
        # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
        lambda_1 = sigmas
        lambda_2 = rhos
        thetas = -thetas

        # Set COV matrix using Lambdas and Theta
        LAMBDA = torch.zeros(B, C, 2, 2, device=thetas.device)  # (B,C,2,2)
        LAMBDA[:, :, 0, 0] = lambda_1
        LAMBDA[:, :, 1, 1] = lambda_2
        Q = torch.zeros(B, C, 2, 2, device=thetas.device)  # (B,C,2,2)
        Q[:, :, 0, 0] = torch.cos(thetas)
        Q[:, :, 0, 1] = -torch.sin(thetas)
        Q[:, :, 1, 0] = torch.sin(thetas)
        Q[:, :, 1, 1] = torch.cos(thetas)
        SIGMA = torch.einsum("bcij,bcjk,bckl->bcil", [Q, LAMBDA, Q.transpose(-2, -1)])  # (B,C,2,2)
        INV_SIGMA = torch.linalg.inv(SIGMA)
        INV_SIGMA = INV_SIGMA.view(B, C, 1, 1, 2, 2)  # (B,C,1,1,2,2)

        # Set expectation position
        MU = (k_size//2) * torch.ones(B, C, 2, device=thetas.device)
        MU = MU.view(B, C, 1, 1, 2, 1)  # (B,C,1,1,2,1)

        # Create meshgrid for Gaussian
        X, Y = torch.meshgrid(torch.arange(k_size, device=thetas.device),
                              torch.arange(k_size, device=thetas.device),
                              indexing='xy')
        Z = torch.stack([X, Y], dim=-1).unsqueeze(-1)  # (k,k,2,1)

        # Calculate Gaussian for every pixel of the kernel
        ZZ = Z - MU
        ZZ_t = ZZ.transpose(-2, -1)  # (B,C,k,k,1,2)
        raw_kernels = torch.exp(-0.5 * (ZZ_t @ INV_SIGMA @ ZZ).squeeze(-1).squeeze(-1))  # (B,C,k,k)

        # Normalize the kernel and return
        mask_small = torch.sum(raw_kernels, dim=(-2, -1)) < 1e-2
        if mask_small.any():
            raw_kernels[mask_small].copy_(0)
            raw_kernels[mask_small, k_size//2, k_size//2].copy_(1)
        kernels = raw_kernels / torch.sum(raw_kernels, dim=(-2, -1), keepdim=True)
        return kernels


if __name__ == '__main__':
    images = torch.rand(10, 3, 100, 100).to('cuda:0')

    Polyblur = GaussianBlurEstimator(n_angles=6, n_interpolated_angles=30, k_size=31, c=0.37, sigma_b=0.40)

    kernels, _ = Polyblur(images)

    print('done')
