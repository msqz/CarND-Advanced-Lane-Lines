import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_frame(image, window_centroids, window_width, window_height):
    def window_mask(width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
               max(0, int(center-width/2)):min(int(center+width/2), img_ref.shape[1])] = 1
        return output

    # If we found any window centers
    if len(window_centroids) > 0:
        warped = image
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height,
                                 warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height,
                                 warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        # add both left and right window pixels together
        template = np.array(r_points+l_points, np.uint8)
        zero_channel = np.zeros_like(template)  # create a zero color channel
        # make window pixels green
        template = np.array(
            cv2.merge((zero_channel, template, zero_channel)), np.uint8)
        # making the original road pixels 3 color channels
        warpage = np.dstack((warped, warped, warped))*255
        # overlay the orignal road image with window results
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()


def find_window_centroids(image, window_width, window_height, margin):
    leftx = []
    lefty = []
    rightx = []
    righty = []
    window_centroids = []

    window = np.ones(window_width)

    from_row = int(3*image.shape[0]/4)
    middle = int(image.shape[1]/2)
    l_sum = np.sum(image[from_row:image.shape[1], :middle], axis=0)
    conv_signal = np.convolve(window, l_sum)
    l_center = np.argmax(conv_signal)-window_width/2

    r_sum = np.sum(image[from_row:image.shape[1], middle:], axis=0)
    conv_signal = np.convolve(window, r_sum)
    r_center = np.argmax(conv_signal)-window_width/2+int(image.shape[1]/2)

    leftx.append(l_center)
    lefty.append(image.shape[0])
    rightx.append(r_center)
    righty.append(image.shape[0])

    window_centroids.append((l_center, r_center))

    for level in range(1, (int)(image.shape[0]/window_height)):
        centroids_pair = [None, None]

        from_row = int(image.shape[0]-(level+1)*window_height)
        to_row = int(image.shape[0]-level*window_height)
        # layer is flattened to be one one row (sum of columns)
        image_layer = np.sum(image[from_row:to_row, :], axis=0)
        # convolve full row
        conv_signal = np.convolve(window, image_layer)
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin, 0))
        l_max_index = int(min(l_center+offset+margin, image.shape[1]))
        # get max of convolved in window frame
        l_max_conv_signal = np.argmax(conv_signal[l_min_index:l_max_index])
        # if there is no signal in current window frame, reuse previous center
        if (l_max_conv_signal != 0):
            l_center = l_max_conv_signal+l_min_index-offset
            leftx.append(l_center)
            lefty.append(to_row)
            centroids_pair[0] = l_center

        r_min_index = int(max(r_center+offset-margin, 0))
        r_max_index = int(min(r_center+offset+margin, image.shape[1]))
        # get max of convolved in window frame
        r_max_conv_signal = np.argmax(conv_signal[r_min_index:r_max_index])
        # if there is no signal in current window frame, reuse previous center
        if (r_max_conv_signal != 0):
            r_center = r_max_conv_signal+r_min_index-offset
            rightx.append(r_center)
            righty.append(to_row)
            centroids_pair[1] = r_center

        window_centroids.append((centroids_pair[0], centroids_pair[1]))

    #draw_frame(image, window_centroids, window_width, window_height)
    return leftx, lefty, rightx, righty


def fit_polynomial(img, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty, left_fit, right_fit


def detect_lanes(img):
    leftx, lefty, rightx, righty = find_window_centroids(img, 50, 80, 100)
    return fit_polynomial(img, leftx, lefty, rightx, righty)
