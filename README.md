[//]: # (Image References)
[title]: ./img/title.png "Title Image"
[result]: ./img/result3.png "Result Image3"
[undist]: ./img/undistort.jpg "Undistorted Chessboard"
[undist_road]: ./img/undistort_road.jpg "Undistorted Road"
[bin_road]: ./img/binary_road.jpg "Binarized Road"
[bird_eye]: ./img/bird_eye.jpg "Bird-eye View"
[bin_bird_eye]: ./img/filtered_bird_eye.jpg "Binarized Bird-eye View"
[hist]: ./img/histo.jpg "Histogram"
[window]: ./img/slide_win.jpg "Sliding Window"
[color_fit]: ./img/color_fit_lines.jpg "Fit Visual"
[eq]: ./img/curv_eq.jpg "Curvature Equation"
[pos_error]: ./img/poss_error.jpg "Possible Error"


# Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![title]

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
---

## Camera Calibration

#### 1. Computing the camera matrix and distortion coefficients:

The code for this step is located in `calibrate.py`

Camera lense is not perfect in capturing the real world in a image. Light can bend too little or too much at the edges. Images thus need to be undistorted before further processing.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![undist]

## Pipeline (single images)

#### 1. Distortion-corrected image.

The image below shows the effect of applying undistortion to the image.

![undist_road]

Although the changes are subtle, the car hood at the two sides in the bottom of the image is clearly unwarped.

#### 2. Binarized Image with Gradient and Color Thresholds

The code for this step is located in `imgfilter.py`

I used a combination of color and gradient thresholds to generate a binary image. Here's an example of my output for this step. 

![bin_road]

The red channels comes from masking the image with gradient thresholds. The green channel is the result of filtering the image pixels in HLS (Hue, Lightness, Saturation) color space, and the blue channel is from filtering in the HSV (Hue, Saturation, Value) space.

More specifically, there are many parameters that can be tuned with gradients given a group of image pixels (size of the group depends on the kernel size). I eliminated pixels outside the specific range in terms of the x and y direction, magnitude of the gradient and angle of the gradient (i.e. horizontal lines).

The thresholds are set in tuning the saturation channel of the image in its HLS space and the value channel in its HSV space. By combining these two channels, the outline of the lane mark can be identify even though the color contrast between the lane marking and the road surface is not that clear.

Here is a snippet of the parameters I could tune for this step of progressing

    KERNEL = 17
    GRAD_X_MAX = 255
    GRAD_X_MIN = 30
    GRAD_Y_MAX = 255
    GRAD_Y_MIN = 30
    MAG_MAX = 150
    MAG_MIN = 50
    DIR_MAX = np.pi/3
    DIR_MIN = 0
    S_MIN = 100          # Saturation channel in HSL
    S_MAX = 255
    V_MIN = 200          # Value channel in HSV
    V_MAX = 255

#### 3. Perspective Tranform into a Bird-eye View of the Road

The function responsible for perspective transform is `bird_eye_view()` in the `calibrate.py` file.

The idea here is that we are taking four points that forms a trapezoidal space in an image, and change our perspective so we see this trapezoid as a rectangle in a new image. Simply speaking, we are transforming a trapezoid into a rectangle. For this we need four points on the original image (src) and the four points of rectangle as the destination (dst).

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 450, 720      | 
| 600, 447      | 450, 0        |
| 685, 447      | 850, 0        |
| 1125,720      | 850, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![bird_eye]

Combined with the image binarization discussed earlier, here is the result:

![bin_bird_eye]

#### 4. Identifying Lane-line Pixels and Create a Polynomial Fit for the Lane Line

The code in this section locates in `laneCapture.py`

To identify the lane pixels, I divide the binarized image into 9 horizontal sections. Because the pixel values either 0 (black) or 1 (white), the section the of the image with the lane line should have the highest sum across any column of the pixels in that particular section.

![bin_bird_eye]

For example, looking at the histogram of the pixel value across each column of the binarized image shown below, the x position at which the graph peaks correspond to that of either of the the lane marks.

![hist]

At each divided section, or window, of the image, the lane section is most likely located near the position of the lane marking at the previous window. We set a margin of search based on the previous location. To find the first segment, we simply locate the local maxima across the bottom-most window of the image.

![window]

Once the x positions of both of the lane are found, we can find a second degree polynomial fit on these positions, producing two functions in the form of `Ay^2 + By + C` for the two lanes.

![color_fit]

#### 5. Calculating the Radius of Lane Curvature and the Distance Away from Center of the Lane

The code in this section also locates in `laneCapture.py`

The curvature of the lanes are calculated using the follow equation.

![eq]

The `A` and `B` corresponds to the coefficients of the lane of fit of the lane. However, the result of the calculation will be in unit pixel, and we still need to convert to metric units.

To do this unit conversion, we can modify the coefficients of the lane of fit as follow:

    Before:
        Ay^2 + By + C

    After:
        mx/(my^ 2)Ay^2 + (mx/my)By + C

The `mx` and `my` corresponds to the ratio of pixels to meter in the x and y direction. For this project the ratio I used are:

    mx = 30/720 # meters per pixel in y dimension
    my = 3.7/400 # meters per pixel in x dimension

The distance that the car is away from the center of the lane is calculated by subtracting the center of the two lanes from the center of the image. The result is in unit pixels so we then multiply by the `mx` ratio mentioned above

#### 6. Result Image

I implemented this step in `advLaneFind.py`

Putting every back together, we can transform the bird-eye view image back into the original perspective after we have identify and drawn the polynomial fit on the two lanes.

![result]

---

### Pipeline (video)

#### 1. Link to Final Video Output. 

Here's a [link to my video result](https://youtu.be/nmufpk-M7qE)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The technique described here is error-prone. The window method of detecting lane line pixels in the binarized image is extremely sensitive to noise in the image. For example, although the model successfully identify the lane mark, the model could easily mistaken the stripe on the left side of the image for the left lane.

![pos_error]

This thus heavily depends on how well the binarization of the image is in isolating the lane marks and the perspective transform stage is in avoiding the amplification of noise. However, it is difficult to tune such model given that that there many situations to account given the degree of the lane curvature, light conditions and positioning of the car. 

Currently, my model does not perform too well on the two challenge videos. The source area of perspective transform is stretched too far out into the center of the image such that the lane marks in a sharp turn will appeared off-center and chopped off after the transform. Secondly, it does not perform well in identifying lane marks in extreme dark conditions.

I can improve the model by reducing the source area for perspective transform and tune the  threshold in both HSV and HLS color space of the image to account for the dark conditions.