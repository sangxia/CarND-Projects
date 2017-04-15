# Advanced Lane Finding

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

[original]: ./writeup_images/r_test2.jpg "Original"
[undistorted]: ./writeup_images/r_ud_test2.jpg "Distortion-Corrected"
[binary]: ./writeup_images/r_bn_test2.jpg "Binary"
[warp]: ./writeup_images/r_wp_test2.jpg "Warped"
[fitwarp]: ./writeup_images/r_wf_test2.jpg "Warped with Fitted Polynomial"
[final]: ./writeup_images/r_final_test2.jpg "Final Result"

## Camera Calibration

This step is finished in the notebook `CameraCalibration.ipynb` (HTML version `CameraCalibration.html`). The chessboard has dimension 9x6, as specified in cell 59. 

I started by preparing "object points", which are the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates (cell 60), and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. This is done in cell 61.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in cell 62. The results are saved to `distort_calibration.pickle` that are used throughout the project. The IPython notebook has some examples of undistorted chessboards. 

## Pipeline

The functions used in the pipeline for both images and videos are in `marklane.py`. Throughout this writeup, I use the following test image as an example.

![Original image][original]

All test image outputs can be found in [here](./output_images/). The filename prefix follows the convention below:
* ud - undistorted version of the image
* bn - threshold binary image
* wp - warped/rectified binary image
* wf - warped binary image with fitted windows and lane polynomials
* final - images with lane areas marked
* summary - a 2x3 grid containing all the above

### Camera Calibration

The method `undistort` applies the result of camera calibration in the previous step. Below is the distortion-corrected image of the test image.

![Distortion-corrected image][undistorted]

### Creating a Threshold Binary Image

The method `binary_lane_threshold` creates a binary image containing pixels that are likely to be lane pixels. The transformation takes into account the following information:
1. The relative magnitude of the gradient in the red channel. For the computation of the gradient, the horizontal direction is given more weight and the vertical direction is given less weight. This is why I set `wy=0.1` in line 56.
2. The direction of the gradients in the red channel. I then filter out pixels whose gradient directions are too horizontal and therefore unlikely to correspond to lane lines.
3. The relative magnitude of the gradient in the S channel in HLS space. This information is processed similarly to the red channel.

The reason I use similar processing methods for both R channel in RGB and S channel in HLS is that sometimes there are more information in one channel than the other. For example, in the test images and `project_video.mp4`, S channel alone is sufficient, but in `challenge_video.mp4`, the lane pixels have very low saturation, so simple thresholding fails completely, and it is therefore important to combine information from other channels.

I also use the following post-processing steps:
1. Denoise with Gaussian blur. This is somewhat effective in getting rid of shadows.
2. Expand the identified pixels to their neighborhoods. This is because the above steps use mostly gradient information to identify pixels, and thus the identified pixels forms the contour of the lane markings. Expanding them to their neighborhoods gives solid lane lines.

The output of this step is shown below.

![Threshold Binary Image][binary]

### Perspective transform

The method `get_perspective_matrix` returns the matrix and its inverse for perspective transformation. The source and destination points are hardcoded. The method `warp_img` applies the perspective transformation using a given matrix.

The following is a warped binary image. The lane lines appear parallel in the warped image so the source and destination points are identified correctly.

![Warped Image][warp]

### Identify the Lane Line Pixels and Fitting a Polynomial

I use a sliding window approach to identify the lane pixels. The method in `marklane.py` is `find_window_centroids`.

A window is a small rectanglular part of the warped binary image. The basic detection strategy is to first identify suitable windows at the bottom of the image. Then, the algorithm moves up level by level and search for the next window in the neighborhood of the current window. 

For each candidate window, I compute a score of its likelihood of capturing lane lines. For the left line, the score of a window is the number of bright pixels in the left half of the window minus the number in the right half, and for the right line, the score is exactly the negation of that. I use this score instead of a simple summation because sometimes the color of the side of a road can be quite similar to the lane markings (as shown in `harder_challenge_video.mp4`), and using this scoring encourages the algorithm to search for windows where transition between road and lane line happens. Windows that score below a certain theshold (specified in `mask_thresh`) are rejected.

To improve the robustness of the algorithm, I keep track of the direction the windows have been moving (in variable `delta`) and a confidence score of this direction (in variable `delta_score`), so that I can move the search in these directions even though no good candidates were found in the current level. This is very helpful when lane markings are not continuous. In the final algorithm, I first search for a new `delta`, and then move both the left and right window in direction `delta`. This guarantees that the windows found by the algorithms are parallel. This simplifies later stages in the pipeline.

After finding the centroids, I do some postprocessing in `fit_lane_centroids`. Observe that in the warped binary image that the left lane line is fully visible, whereas the right lane line contains relatively few pixels. In this step, I fill in the windows for the right line using the knowledge that the lines should be parallel. This step is implemented before I start to keep track of direction in the previous step, and is now not strictly needed.

The method `fit_lane_poly` fits a polynomial using the window centroids identified above. The method returns an array of 4 coefficients, and a confidence score determined by the number of pixels that fall in the neighborhood of the fitted polynomial line. Note that although there are 3x2=6 coefficients for two quadratic polynomials, because of the guarantee above that the lines are parallel, the coefficients for the quadratic and linear term of both lines are identical. I also perform the following sanity checks:
1. If the width of the lane changes suddenly or is not reasonable, reject the lane lines that are responsible for the sudden change (line 306-317).
2. For videos, allow large change in curvature only if the confidence score is high (line 324-333). Otherwise decrease the amount of update applied in this step. 

For videos, I also applied smoothing so that lanes do not jitter too much.

The result of this step is plotted with `fit_warped` as shown below.

![Warped Image with Fitted Polynomial][fitwarp]

### Curvature and Vehicle Position

The formula for curvature calculation is based on the code in the course material. I improved it by noticing that there is a direct relationship between the polynomial in pixel space and the polynomial in real world space. The coefficients in pixel space can be converted easily and there is no need to fit another polynomial. This is implemented in method `curvature`.

To calculate the position of the vehicle with respect to lane center, I simply measure the horizontal pixel distance between the center of the image and the center of the predicted lane lines, and then scale it to real world distance. This is done in line 390-392.

### Final Result

The method `draw_lines` takes the undistorted image, the coefficients of the fitted polynomials, and the inverse perspective transform matrix and plots the lane area back down to the road. The final result is as follows.

![Final Result][final]


### Video

1. [Project video](./output_project_video.mp4): the marked area tracks the actual lane area quite well throughout the video. The markings further away from the vehicle overreacts to turns occasionally but are quickly corrected. The reported statistics are within reasonable range.
2. [Challenge video](./output_challenge_video.mp4): the left edge of the marked area tracks the left lane line quite well throughout the video, whereas the right edge expands a bit to the next lane for most of the video (the near side is not too bad, the further away from the vehicle the worse). Given that in my approach the fitted polynomials are guaranteed to be parallel, I find this hard to explain. One explanation is that my perspective transformation is a bit off, but I do not find it very satisfactory. Another challenge of this video is that there is a color change on the road right in the middle of the lane, and sometimes there are diamond shaped markings on the road. The algorithm seems to have dealt with it well.
3. [Harder challenge video](./output_harder_challenge_video.mp4): performed reasonably well for about 2/3 of the video, dealt with adversarial lighting condition quite well. Around second 8, was slow to adapt to the sharp turn. Second 25-35, got distracted by coming vehicles in the opposite lane, which is quite dangerous. Failed the final sharp turn completely.

## Discussion

I have already touched on some of the issues. Here I summarize them and list some of the other problems that I encountered, and possible solutions and improvements.

### Color Space

Initially, I used color gradient for grayscale image and a simple thresholding on the S channel. This worked well until I tested it on `challenge_video.mp4`, where the lane line pixels have very low saturation. This is why I switched to using gradient for S channel as well. This could still fail but my understanding of color theory is not good enough to identify exactly how. I think one alternative would be to run parallel pipelines on both the grayscale image and the S channel (or possibly more channels) and then score and combine the results.

### Noise

In the current pipeline, the threshold for the color gradients are quite relaxed, because otherwise the actual lane pixels get filtered out. The consequence is that the initial binary images are quite noisy. Applying Gaussian blur helped in this respect.

When identifying the lines using sliding windows, the simple algorithms gets distracted by noise caused by things such as shadows and objects in adjacent lanes. The problem is exacerbated when the lane lines are not solid. The situation improved a lot after I implemented a feature that tracks the direction of the detected windows, and forces the algorithm to detect parallel windows. The next step would be to incorporate lane predictions from previous frames to further help the algorithm.

### Unusual Lane Location

One reason the algorithm completely failed towards the end of the harder challenge is that the left lane starts almost on the center right of the frame, and the right lane is sometimes invisible in the frame. My current implementation searches for the *first* window for the left lane only in the left half of the frame, and the *first* right window in the right half. One solution would be to use previous predictions to enable a more dynamic search range.

### Speed

The algorithm currently processes about 3 frames per second, which is not very fast. Much time is spent on calculating the gradients for the whole frame in various channels. One could use previous predictions to define a narrower search range, but I am not sure how to make OpenCV only compute gradient in a specific range.

