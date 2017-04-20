# Vehicle Detection Project

The goals / steps of this project are the following:

* Extract features such as Histogram of Oriented Gradients (HOG) and binned color histogram to train a classifier using a labeled training set of images
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Most components of my vehicle detection pipeline are in `utils.py`.

[test]: ./writeup_images/resized/test4.jpg
[sliding]: ./writeup_images/resized/raw_boxes_test4.jpg
[heatmap]: ./writeup_images/resized/hm_no_thresh_test4.jpg
[thresh-heatmap]: ./writeup_images/resized/hm_thresh_test4.jpg
[final]: ./writeup_images/resized/final_test4.jpg

## Histogram of Oriented Gradients (HOG)

The code that extracts features from training data can be found in the file `prepare_features_cv2.py`. Throughout the project, I use the HOG feature extractor from OpenCV instead of skimage because my experiment shows that it is much faster. In fact, my earlier solution for this step using skimage took about 105 seconds, but the current OpenCV based solution took less than 15 seconds.

The parameter setting is in line 9 of the file. I took the recommended default parameters, and I found the performance to be very good out of the box.

The for loop in line 27-29 extracts features for the 64x64 images. The `get_features_cv2` method in `utils.py` extracts binned histogram feature for the Hue channel, as well as the HOG feature vectors for a number of other channels. Later I experimented with feature selection so I didn't end up using all of them.

One thing to note is that when extracting features, I first resize the image from 64x64 to 66x66, and extract HOG feature for the 64x64 square at the center. This is because gradient computation on the border can cause problem, and I think it is for this reason that the efficient HOG subsampling procedure in the lessons doesn't work well with other parts of the pipeline without some extra effort.

## Vehicle vs. Non-vehicle classifier

This step can be found in the files `validation.py` and `validate_on_crop.py`. I use the first one to guide feature and hyperparameter selection, and the latter one to further test my model on images cropped directly from `test_images/`, which can be found in `cropped_test_images/`. 

The `run` function in `validation.py` takes as input the data and parameters (including the features to use, and the C and gamma parameters for SVM classifier). It first use a `StandardScaler` to normalize the features, and then trains a classifier on the training dataset, followed by an evaluation on the validation dataset. 

I tried various methods of splitting train and validation set: training on GTI dataset and predicting on the remaining, and vice versa, random split, and a stratified split --- that is, taking the first 70% of each category as training set and the rest as validation. An interesting observation is that the GTI vehicle images are mostly from behind, so it doesn't generalize very well to the KITTI dataset which is more diverse. 

For feature selection, I tried combinations of HOG from different channels with and without the Hue histogram features. I did a grid search for C and gamma. Since sklearn's implementation of SVM classifier is single-threaded, I further used `joblib` to parallelize the search process (line 99).

The validation accuracy of the saved SVM model was around 0.995. It achieved similar accuracy on both vehicle and non-vehicle images.

## Sliding Window Search

The sliding window search functionality is implemented in `utils.py`. There are three main functions: `detect_vehicles_from_crops`, `detect_vehicles_parallel` and `detect_vehicles_in_boxes_parallel`. The latter two basically just generates parameters and then calls the first function, so I will focus on the first one.

The function `detect_vehicles_from_crops` takes as input an image in BGR format, the StandardScaler and the classifier, as well as a list of crops. Items in the list of crops describes on which part of the image and at what scale we should perform the sliding window search. 

The function itself is straightforward: crop and scale the images as needed, then compute the features. Much of the complexity of its structure is due to the effort to make the search run faster and in parallel. First, a list of parameters describing each sliding window is generated (line 81-89). Then, feature extraction for each sliding window is performed (line 90-94). The results are transformed into a numpy array, and then divided into large chunks sent to the classifier. This is done so as to minimize overhead.

For vehicle detection on images, one can use the `detect_vehicles_parallel` function. The crop list defined in this function (line 118-126) performs sliding window search on different parts of the image at different scales. Small windows are only applied to the more distant part of the images.

### Optimization

* Restricting the search to only parts of the images. For example, in `detect_vehicles_parallel`, small scale search is only performed in the middle stripe of the image, where vehicles tend to be far away and thus small.
* Parallelization. On single images, single-thread sliding window search takes 8 seconds, parallelized window search using `joblib` reduces this to 3 seconds on a 4-core machine.
* In addition to using thresholds on heatmaps to reduce false detections, I also verify that the boxes that I found are of reasonable shape (line 222-225 of `utils.py`). Tall or flat boxes are rejected.

### Example

To illustrate each step of my pipeline, let us now look at an example. Here is the image:

![Input test image][test]

A sliding window search at multiple scales gives many overlapping bounding boxes:

![Initial result of sliding window search][sliding]

The region covered by these boxes is shown below:

![Covered region][heatmap]

The region of the two cars are connected in the diagram above, making it difficult to draw individual bounding boxes for the two cars. We can apply a threshold and only consider pixels that are covered by multiple boxes. The pixels between the cars are thus filtered out.

![Thresholded heatmap][thresh-heatmap]

We see that there is still a thin slice of highlighted pixel in between the two cars, likely caused by the particular setting of the strides of the sliding window. This is filtered out by the check on box sizes mentioned above. 

![Final result][final]

## Video Implementation

Here's a [link to my video result](./output_project_video.mp4) The video is generated using code `mark_videos.py`.

### Video-Specific Optimization

* Instead of performing a sliding window search on the entire lower-half of the frames, only search around the left and right border of the frame (which is where new vehicles come in), and around regions where vehicles have been detected in the previous frame. This improved the detection performance to about 1.8 seconds per frame, compared to 3 seconds for detection in single image.


### Filtering of False Positives and Combining Bounding Boxes
I use a variable `states` to keep track of the relevant parameters and previous detections. To get the bounding boxes for a new frame, I first run the pipeline for the images, obtain the heatmap and bounding boxes for the new frame. I then devised an algorithm that matches the new bounding boxes with those from the previous frames based on their area of intersection. The result is then smoothened to produce stable predictions for bounding boxes. This merging step is done in `merge_detections`. The for loop in line 18-31 tries to match previous predictions with new predictions. For the new boxes that do not match previous predictions, I record them and wait for the next few frames to see if they are detected again. Only boxes that have been detected in a few consecutive frames are drawn on the final output.

I also experimented with integrating heatmaps over frames, but I don't find it to be more robust than my current solution.

### Evaluation of Video Output

Overall the detection worked well for the project video. A frame-by-frame description of the key information of the detection process can be found in the file `log_project_video.txt`. 

Only one frame (frame 1039) contains a false positive detection. From the log, we can see that it was detected over several frames, but the above filtering algorithm eliminated it to a large extent.

It also happened a few times where the sliding window search did not return any bounding boxes for a particular frame, but this one-time issue effectively smoothed over by the above algorithm.

## Discussion

I now discuss some of the issues / limitations of my pipeline.

* Speed. Despite all the efforts in optimizing the algorithm, the detection is still far from real-time. I don't really know how to dramatically improve the performance with a CPU-based solution. However, given that most of the pipeline is easily parallelizable, I expect a significant performance gain using the same pipeline on a GPU.

* Overlapping vehicles. The current algorithm for drawing bounding boxes is very simple: look at the regions marked by `scipy.ndimage.measurements.label()` and find the box that encloses the whole region. When images of vehicles overlap, this results in a single large box that covers both vehicles. Perhaps more sophisticated / computationally intensive algorithms can handle this.

* Front of the vehicles, and vehicles in the opposite directions. The GTI dataset mostly captures the rear of vehicles, so the classifier trained on it does a better job detecting rear than front. This is not ideal because when vehicles enter the frame we typically see their front first. To improve this, we can try to collect more diverse images, as well as data augmentation.

* Detection near the boundary. Looking at the output video, one notices that the bounding boxes never extend to the boundary of the frame. I think this is because pixels near the boundaries are covered by fewer sliding windows, so they tend to get filtered out when we apply a threshold on a heatmap.

* Vehicles far away. For efficiency reasons, I do not run sliding window search on vehicles that are far away. This would not be difficult to implement --- simply add the appropriate parameters to `crop_list` when calling `detect_vehicles_from_crops`.

* Classifier performance. Through validation, I decided on a model that achieved about 0.995 accuracy on the given dataset. However, given that we are making a lot of predictions over many frames, even a 0.5% error rate can be problematic. Going through `log_project_video.txt`, I can see that there are quite a few instances of false positives and negatives. Most of them are filtered so they did not affect the final output too much, but I think it would be beneficial to collect these failures and train a better classifier.

