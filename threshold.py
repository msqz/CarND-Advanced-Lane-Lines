import numpy as np
import cv2
import helpers


def by_color(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    binary = np.zeros_like(s_channel)
    h_range = (h_channel >= 12) & (h_channel <= 27)
    s_range = (s_channel >= 80)
    l_range = (l_channel >= 80) & (l_channel < 200)
    binary[h_range & s_range & l_range] = 1
    return binary


def by_gradient(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    sobel = np.uint8(255*sobel/np.max(sobel))

    binary_mag = np.zeros_like(sobel)
    binary_mag[sobel >= 60] = 1

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    graddir = np.arctan2(abs_sobely, abs_sobelx)

    binary_grad = np.zeros_like(graddir)
    binary_grad[(graddir >= 0.2) & (graddir <= 1.4)] = 1

    binary = np.zeros_like(graddir)
    binary[(binary_mag == 1) & (binary_grad == 1)] = 1
    return binary


def to_binary(img):
    color = by_color(img)
    gradient = by_gradient(img)
    binary = np.zeros_like(color)
    binary[(color == 1) | (gradient == 1)] = 1

    return binary
