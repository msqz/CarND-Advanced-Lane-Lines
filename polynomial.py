import numpy as np
import cv2
import matplotlib.pyplot as plt

sync_each = 5


def find_lane_pixels(img):
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    nwindows = 9
    margin = 110
    minpix = 40

    window_height = np.int(img.shape[0]//nwindows)
    nonzero = img.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        # left window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        # right window
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]

        good_right_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(img, left, right):
    if not left.detected or not right.detected:
        print('undetected')
        left.allx, left.ally, right.allx, right.ally = find_lane_pixels(img)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    if len(left.allx) != 0:
        left_fit = np.polyfit(left.ally, left.allx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

        left.detected = True
        left.recent_xfitted.append(left_fitx)
        left.bestx = np.mean(left.recent_xfitted[-sync_each:], axis=0)
        left.current_fit = left_fit
        left.recent_fit.append(left_fit)
        left.best_fit = np.mean(left.recent_fit[-sync_each:], axis=0)
    else:
        left.detected = False

    if len(right.allx) != 0:
        right_fit = np.polyfit(right.ally, right.allx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        right.detected = True
        right.recent_xfitted.append(right_fitx)
        right.bestx = np.mean(right.recent_xfitted[-sync_each:], axis=0)
        right.current_fit = right_fit
        right.recent_fit.append(right_fit)
        right.best_fit = np.mean(right.recent_fit[-sync_each:], axis=0)
    else:
        right.detected = False


def search_around_poly(img, left, right):
    margin = 100

    nonzero = img.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]

    left_lane_inds = ((nonzerox > left.best_fit[0]*nonzeroy**2 + left.best_fit[1]*nonzeroy + left.best_fit[2] - margin) &
                      (nonzerox < left.best_fit[0]*nonzeroy**2 + left.best_fit[1]*nonzeroy + left.best_fit[2] + margin))
    right_lane_inds = ((nonzerox > right.best_fit[0]*nonzeroy**2 + right.best_fit[1]*nonzeroy + right.best_fit[2] - margin) &
                       (nonzerox < right.best_fit[0]*nonzeroy**2 + right.best_fit[1]*nonzeroy + right.best_fit[2] + margin))

    left.allx = nonzerox[left_lane_inds]
    left.ally = nonzeroy[left_lane_inds]
    right.allx = nonzerox[right_lane_inds]
    right.ally = nonzeroy[right_lane_inds]

    fit_polynomial(img, left, right)
