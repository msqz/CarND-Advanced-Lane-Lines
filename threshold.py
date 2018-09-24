import numpy as np
import cv2
import helpers


def by_color(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    binary = np.zeros_like(s_channel)

    # yellow
    lower_yellow = np.array([12, 40, 100])  # 100
    upper_yellow = np.array([27, 200, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    # white
    lower_white = np.array([0, 194, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    binary = yellow_mask + white_mask

    return binary


def by_gradient(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    sobel = np.uint8(255*sobel/np.max(sobel))

    binary_mag = np.zeros_like(sobel)
    binary_mag[sobel >= 70] = 1

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    graddir = np.arctan2(abs_sobely, abs_sobelx)

    binary_grad = np.zeros_like(graddir)
    binary_grad[(graddir >= 0.2) & (graddir <= 1.4)] = 1

    binary = np.zeros_like(graddir, dtype=np.uint8)
    binary[(binary_mag == 1) & (binary_grad == 1)] = 1

    return binary


def to_binary(img):
    color = by_color(img)
    gradient = by_gradient(color)

    return gradient
