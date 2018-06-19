"""
Capture Lane Lines by Combining Different Gradient Thresholds
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

KERNEL = 17
GRAD_X_MAX = 255
GRAD_X_MIN = 10
GRAD_Y_MAX = 255
GRAD_Y_MIN = 10
MAG_MAX = 150
MAG_MIN = 50
DIR_MAX = np.pi/3
DIR_MIN = 0
S_MIN = 100
S_MAX = 255
V_MIN = 200
V_MAX = 255

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return mag_binary

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, int(orient == 'x'), int(orient != 'x'))
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))  
    grad_mask = np.zeros_like(scaled_sobel)
    grad_mask[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_mask

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) 
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    gradDirect = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # 5) Create a binary mask where direction thresholds are met
    dir_mask = np.zeros_like(gradDirect)
    # 6) Return this mask as your binary_output image
    dir_mask[(thresh[0] < gradDirect) & (gradDirect < thresh[1])] = 1
    return dir_mask

# Thresholds the S-channel of HLS
def s_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_mask = np.zeros_like(s_channel)
    s_mask[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_mask

# Thresholds the V-channel of HSV
def v_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hls[:,:,2]
    v_mask = np.zeros_like(v_channel)
    v_mask[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1
    return v_mask

def binarize(image, kernel=KERNEL, x_thresh=(GRAD_X_MIN, GRAD_X_MAX),
             y_thresh=(GRAD_Y_MIN, GRAD_Y_MAX), magni_thresh=(MAG_MIN, MAG_MAX),
             dir_thresh=(DIR_MIN, DIR_MAX), sat_thresh=(S_MIN, S_MAX),
             val_thresh=(V_MIN, V_MAX)):
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=KERNEL, thresh=x_thresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=KERNEL, thresh=y_thresh)
    mag_binary = mag_thresh(image, sobel_kernel=KERNEL, thresh=magni_thresh)
    dir_binary = dir_threshold(image, sobel_kernel=KERNEL, thresh=dir_thresh)
    sat_binary = s_thresh(image, thresh=sat_thresh)
    val_binary = v_thresh(image, thresh=val_thresh)

    grad_binary = np.zeros_like(mag_binary)
    grad_binary[ ((gradx == 1) & (grady == 1)) | 
                 ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Combining thresholds
    combined = np.zeros_like(mag_binary)
    combined[ (grad_binary == 1) | (sat_binary == 1) & (val_binary == 1) ] = 1

    # hightlight contributions from each channel
    channels = 255*np.dstack((grad_binary, sat_binary, val_binary)).astype('uint8')
    return combined, channels