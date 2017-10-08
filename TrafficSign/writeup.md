# Traffic Sign Recognition


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: extra/1e.jpg
[image2]: extra/2e.jpg
[image3]: extra/3e.jpg
[image4]: extra/4e.jpg
[image5]: extra/5e.jpg

## Dataset Exploration and Visualization

My code and results for dataset exploration can be found in the section titled "Step 1" in the notebook. I used pandas and matplotlib to calculate summary statistics and visualize random samples of the images.

* The pickled dataset traffic-signs-data.zip contains three sets of images: the training set has 34799 images, the validation set has 4410 images, and the test set has 12630 images.
* All images are 32x32 with 3 channels.
* There are altogether 43 classes. Class distribution can be found in the notebook.

For visualization, I wrote function `plot_random` that samples a random set of images and displays them together with their names. I also plotted the class distribution. We see that the train/validation/test ratio varies quite a bit for certain classes.

## Design and Test a Model Architecture

### Preprocessing

For each image, I scaled the data such that it has mean 0 and standard deviation 1. After that, I clipped the values at +/- 7.5. The value is determined by some trial and error. Both the scaling and clipping are to avoid extreme input values and make the training more stable. The whole normalization step is implemented in the function `data_normalization`.

### Training data augmentation

In order to encourage the model to learn features that generalize to unseen data, I used image rotation and shifting to generate additional training data.  I used some functions from `tensorflow.image.contrib` and implemented functions `rotate_op` and `shift_op`. 


### Model Architecture

The notebook contains a diagram of the model exported from tensorboard.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Rotate and shift  | Data augmentation |
| Convolution 3x3 and 5x5	| 1x1 stride, same padding, outputs 32x32x16 for 5x5 convolution, 32x32x8 for 3x3 convolution, both followed by ReLU and then concatenated into 32x32x24 	|
| Convolution 5x5 | 1x1 stride, same padding, output 32x32x48 |
| ReLU | |
| Maxpool | 2x2 size, 2x2 stride, valid padding |
| Convolution 3x3 | 1x1 stride, same padding, output 16x16x96 |
| ReLU | |
| Maxpool | 2x2 size, 2x2 stride, valid padding |
| Dropout | |
| Fully connected | Output 1024 |
| ReLU | |
| Dropout | |
| Fully connected		| Output 43 |
 
The outputs of the final layer are the logits. 

### Training Details

#### Data Shuffling

Since the images of each sign are from successive frames in videos, neighboring images are quite similar. It is therefore very important to shuffle the data.

#### Regularization

In addition to using dropout, I also added l-2 weight regularization. The importance factor is chosen by trial and error. Small changes can lead to big differences in model performance.

#### Sampling of Rare Classes

  I noticed that earlier models were not performing well on certain classes, so I decided to introduce more samples from these classes. The classes are defined in variable `weak_labels`. Batches of images from this set are combined with the usual batches. One parameter that needs to be tuned is the relative size of the two batches.

#### Parameters

I used adam optimizer, with batch size 1024 for the usual batch and 24 for the weak class batch, so the actual batch size is 1048. The model was trained for 30 epochs, with a learning rate of 0.001 and a factor 0.001 for l-2 regularization. I trained an ensemble of 3 models.

The dropout rate was set to 0.5 for the first 10 epochs and then 0.2 for the remaining 20. One motivation behind this is the observation that training accuracy quickly reached 1 when validation accuracy was still around 0.95, so by decreasing dropout in later stages, it may force the model to find new features that can be helpful.

### Evaluation

As shown in cell 124 and 126, the ensemble of 3 models achieved a validation accuracy of 0.981 and a test accuracy of 0.9848. 

I also plotted the classification performance of each classes as a confusion matrix. It is interesting to see that there are some discrepancies in performance for certain classes. I think this is most likely due to the fact that those classes are very small, and thus random variation can have a huge impact on accuracy.

### Process

I started with a LeNet-like architecture. After experimenting with kernel size and introducing dropout and l-2 regularization, I was able to achieve 0.95 accuracy. I then experimented with data augmentation techniques and different dropout rates.  The confusion matrix was very helpful in helping me understanding what kind of problems my models were having, and generally pointed in the direction of adding convolutional layers and increasing the size of the feature maps. 

I found that accuracy can vary by as much as 0.01 for the same architecture if we just run the same training again.

### Test on New Images

Here are five German traffic signs that I found on the web:

![Round Pedestrain][image1] ![Pedestrain, unusual angle][image2] 
![Unseen class][image3] ![End of no passing][image4]
![Weight limit][image5]

Finding traffic sign photos of a given class seems to be the most challenging task in this project. I collected images from classes where the model did not do well, or images that are non-standard, in order to see what the model would do in these circumstances.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Round Pedestrain | Speed limit (50km/h)   									| 
| Pedestrain, unusual angle     			| General Caution 										|
| Merging from right with priority					| Right-of-way at the next intersection											|
| End of no passing	      		| Dangerous curve to the right |
| Vehicles over 3.5 tons prohibited |  Vehicles over 3.5 tons prohibited|

The fifth one is correct. The third class is not in the dataset, but the shape is very similar and the prediction is correct. For the first two images, the image and the model prediction has common shape and color, so one direction for improvement could be to distort or zoom the images as a way of input data augmentation. The model was very confident in all its predictions, as noted in the notebook. The top 5 classes and their probabilities are in cell 136 and 137.
