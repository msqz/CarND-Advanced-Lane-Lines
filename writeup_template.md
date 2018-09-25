## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibrated.jpg "Calibrated"
[image2]: ./output_images/undistorted.jpg "Undistorted"
[image3]: ./output_images/binary.jpg "Binary"
[image4]: ./output_images/transformed.jpg "Transformed"
[image5]: ./output_images/lines_detected.jpg "Detected lines"
[image6]: ./output_images/output.jpg "Output"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

That step is performed by `calibrate()` function in `transformation.py` file (line 10). The input is composed of files from `camera_cal` directory. The function returns transformation matrix and distortion coefficients. Those values are used for undistorting images in the next stage of the pipeline.

The idea is to match the real chessboard corners (`objpoints` array) with corners detected on the captured image (`imgpoints`). Corners detection is done by `cv2.findChessboardCorners()` function. Processing all of the calibration images, the `objpoints` and `imgpoints` arrays are passed to `cv2.calibrateCamera()` function, which calculates the mapping between real and distorted points. That mapping is the transformation matrix and the array of distortion coefficients

I've used `cv2.cornerSubPix` function (https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#cornersubpix) to increase the accuracy of corners detection.

Camera calibration is performed only once, before the pipeline.

Below is the result of undistoring test image with `cv2.undistort()`, using calculated transformation matrix and distortion coefficients.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the transformation matrix and distortion coefficients from camera calibration step, I call the `cv2.undistort()` function to undistort each frame of the input video (it's in `undistort()` function in file `transformation.py`, line 34).

Below is the example frame after undistorting:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (functions `by_color()` and `by_gradient()` in file `threshold.py` - lines 6 and 27). Color detection is based on filtering yellow and white elements of the image. Gradient thresholding is based on calculating magnitude and direction of gradient. Here's the image used in previous step, but now after mentioned transformations:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transformation takes place in function `warp()` in file `transformation.py` (line 43).
The source and destination points are defined in variables (respoctively) `warp_from` and `warp_to` in `transformation.py` file:

```python
warping_from = np.float32([[200, 720], [604, 450], [696, 450], [1120, 720]])
warping_to = np.float32([[200, 720], [200, 0], [1120, 0], [1120, 720]])
```

 The image below shows that after warping lines are parallel:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Detecting lanes is based on sliding window and searching from prior frame. When the processed frame is the first one OR the detected lines are not valid, then sliding window algorithm is executed. Otherwise searching is done in margin around the lines detected in previous frames (file function `pipeline()` in file `main.py`, lines 75 - 78).

The x-axis in the picture below is scaled up to 1640px. That's because I'm using extended canvas for drawing the lines, it's explained in step 6. of the report.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius is calculated in method `get_curverad()` of the `Line` class (file `line.py`, line 26). It's the result of radius formula, shown in the lectures, multiplied by pixel to meters scale (defined in file `line.py`, lines 3-4).

Offset from the center of the lane is calculated function `pipeline` in file `main.py` (line 82).
It's the difference between the left and right edge of the lane, scaled using pixel to meters scale. Negative offset means being more to the left side of the lane, positive - to the right side.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The polynomial function graph would be cropped when results are out of the 0 - 1280 range (image width). That occurs in case of a sharp turn. To overcome that I first extend the canvas on which the functions will be plotted to be `3 x height` and `3 x width` of the original image (line 28 of function `expand()` in file `drawer.py`). Then I move the points of the function graph to the center of expanded canvas (lines 31-31 of function `expand()` in file `drawer.py`). That way the graph can be plotted for every value of y (0 to 720), even when it exceeds the edge of the original image (polynomial(y) < 0 or polynomial(y) > 1280). Then the expanded canvas is warped back to be in perspective of the camera image (line 77 in function `draw_lane()`) and then cropped to fit into the original image (line 78, 79).

The output image is built in function `combine` in file `drawer.py`. The image contains preview of processing stages, so the correlation between processing result and the output can be seen.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline has some minor problems with detecting small broken lines on curves. It's a matter of avoiding noise when detecting lines - random bright spots interfere the measeurements, so they have to be eliminated.

I think there are 2 main areas for improvement:
1. Harsh light - the dynamic range of the image is so wide, that it's hard to detect a lane when one of edges is in the shadow and the other is in bright light. I'm wondering about separating the shadow areas and normalize the brightness level there. In harsh ligting shadows have very sharp edges and they are very dark, comparing to the rest of the scene, so maybe they can be detected using some gradient processing.

2. Tarmac surface - smooth and dark tarmac with bold lines makes detection easy, but the bright one with or the one with patches creates a lot of noise. That could be handled by detecting shapes which are not similar to road lines, and by eliminating them from the image.