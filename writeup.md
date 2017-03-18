# Finding Lane Lines on the Road

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[roi]: roi.png "Region of Interest"

![Grayscale][image1]

### Reflection

### 1. Pipeline

My pipeline consisted of the following steps.

First, I converted the images to grayscale and added some Gaussian blur to remove small noise in the image.

Then, for all pixels in the grayscale image whose brightness is below a certain threshold, I set their brightness
to zero. This is because in difficult situations such as complicated shadows on the roads, edge detection returns
very complicated patterns, making it very difficult to detect lane lines. Lane markings are usually bright so they
are preserved in this step.

After that, I run the Canny edge detection algorithm to highlight the edges in the image. This is followed by
a region of interest masking. The region of interest in my algorithm is roughly a trapezoid.
![Region of Interest][roi]

Then, I use Hough transformation and line detection algorithm to get a set of line segments.
In order to draw a single line on the left and right lanes, I modified the draw_lines() function as follows:
1. I added a collect_lines() function that returns the estimated single line for the left and right lane lines.
The function first sorts all line segments into left lane, right lane, or discard them according to their location
and the slope of the line segments. In particular, segments that are vertical or horizontal are discarded.
This is then followed by taking sample points from the remaining line segments, and use sklearn's linear models
to find a best fit.
2. To mitigate jittering of lane lines in the video caused by vibration of the vehicle as well as imperfection of the
algorithm, the draw_lines() function keeps a copy of the lane lines drawn earlier. The estimated lane lines from step 1
is then averaged to produce a smoothed lane line.


### 2. Potential shortcomings

The lane lines are fitted with straight lines. This means that on a curved road, the predicted line will not fit well.
I have experimented with fitting the lines with higher degree polynomials, but it is very hard to regularize and the 
result was not satisfactory.

Another issue that can be seen from the produced videos is that for broken lane lines the detection is not as stable
as that for the solid lane lines. This is particular noticeable when the nearer part of the lane line is missing. 

The algorithm also does not behave well in adversarial lighting conditions. This can be seen from the challenge.mp4 video,
where the vehicle enters a particularly bright road segment.
The predicted lines do not always track the actual lines closely. This indicates instability in the algorithm.
One potential cause is that the threshold settings for various part of the algorithm do not work well for this situation.
Another source of instability could be that, in the current setting, the min_line_length for Hough detection is a rather
small number, partly due to the fact that for broken lane lines, sometimes the only available line segments are far and 
therefore small. It is possible that a more systematic search of parameters can return better results.

There are other scenarios that we have not tested on. For example, there could be a problem if we are somewhat closely 
following another car and much of the lane lines are not visible.

### 3. Possible improvements 

One potential improvment to my pipeline is to perform some transformation on the image before doing Hough line detection.
For example, we can first stretch the region of interest into something of a more rectangular shape. Visualization of
intermediate steps of the current pipelines shows that it is very hard to extract useful information for regions that are
far away, and stretching the far-away part of the image may help separate the actual lane lines from noise.

For the problem of curved lanes, I think we need a more powerful model than simple line detection. We can for instance 
try to classify which part of the image represents part of a lane line, and then connect them together. Before I adopted
the linear regression approach, I experimented with a simpler version of this idea using line segments from Hough detection, 
but sometimes there are too many irrelevant segments to make this useful, and smoothening across frames is also tricky.

As for lighting conditions, a natural thought is to use a more adaptive algorithm. But that can be computationally 
expensive. Also, this may not be necessary with a more advanced lane line detection model.

