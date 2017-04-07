# Behavioral Cloning

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Summary of Project Files

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model-16.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run1.mp4, run1r.mp4, run2.mp4, run2r.mp4 containing recordings of the model driving autonomously on both tracks in both directions

## Training Strategy

### Data Collection

I recorded two laps in each direction on each track driving normally (center of the road/lane, smooth in the corner). Due to the complexity of track two, I drove at a much lower velocity, and changed lanes somewhat frequently. 

I collected a total of 14167x3 images coming from all cameras. As I discuss below, I used the left and right camera images in order to teach the model to recover from sub-optimal positions, and that proved to be sufficient.

The original images are of dimension 160x320x3. Before feeding them into my model, I cropped them vertically to 80x320x3 to a smaller region of interest in order to help the model focus on the important parts of the images, and also to speed up training. After that, for each image, I centered the image by subtracting the mean and dividing by the standard deviation of the pixels, and clipped the results into range [-10,10].

### Training Methods

I set aside 10% of the center camera image as a validation set. The steering angles for track two are much larger than those in track one, thus to get a better understanding of the training process, I actually had two separate validation set, one for each track. For those frames whose center image is included in the validation set, I excluded their left and right camera image from the training set. 

During training, the training set is first shuffled, and then fed into the model with a batch size of 128. The model also receives a number for each image specifying which camera the image came from. To help reduce overfitting and generate more training data, I randomly flipped each image horizontally, and made necessary adjustments in the training data.

I used an adam optimizer and mse as loss. I found that periodically decreasing the learning rate led to lower validation loss. There does not seem to be a strong correlation between validation loss and performance in driving (for loss below a certain level). I used dropout and l2-weight regularization to reduce overfitting and improve generalization. I trained the model for 40 epochs.

## Model Architecture

My model consists of 4 convolution blocks, followed by a Dropout and a Dense layer. Each convolution block consists of two stacked convolutional layer of the same kernel and output size, followed by a max pooling layer. The filter sizes are 3x3 for the first layer and 5x5 for the second to fourth layer, and depths increases from 16 to 128. In my final model, the last convolutional block is also concatenated with features from the previous convolutional block downsampled using max pooling. 

I made two important adjustments to the basic architecture described above.

1. I recorded the training data by steering with a mouse in order to achieve smoother, less dramatic steering. A consequence of this is that most steering angles (in the driving log csv files, the recorded number is in fact the fraction of the maximum steering angle) are very small. This caused some numerical issues and made it difficult to train. Therefore I multiplied all steering angles with a common factor (10 in my implementation). Correspondingly, in `drive.py`, I reduced the model output by a factor to get the actual steering angle. For reasons described below, this compensation factor is not 1/10=0.1 but slightly larger.

2. When feeding the model with images collected from the side cameras, the steering angles need to be adjusted. Early on, I tried adding or subtracting the steering angle by a fixed number, and was able to train models to drive on both tracks. However, this adjustment factor is different for each track and needs to be hand tuned. Using some trigonometry, I worked out the exact formula for this adjustment. There is one parameter that depends on the physics of the vehicle that is unknown. I decided that instead of hand-tuning it, I can simply make it part of the model and train it. The parameter is in the `side_factor` layer. This approach seems to have worked out well.

### Modification to `drive.py`

The above is sufficient for track one. However, I noticed several issues for track two and made some changes to the driving module accordingly.

1. The track has steep uphill and downhill roads, and slope changes rapidly. When there is a steep uphill followed immediately by a level section, the vehicle may leave the ground if driving at full speed, making steering impossible. I changed `drive.py` to limit the speed to 15 mph. I haven't spent much time figuring out how to control brakes, so when going downhill the vehicle can still gain maximum speed. On steep downhill sections with many zigzags this can cause problem.

2. Sometimes the vehicle seems to steer a little too late. My hypothesis is that steering is not updated at sufficient frequency. To compensate for this, I amplified the final steering by 30%. In simulation, this helped a lot in sharp turns without sacrificing too much in terms of driving stability on the straight sections of the tracks.

## Simulation results

The vehicle was able to drive around both tracks in both directions without leaving the road. Moreover, when applying perturbations, the vehicle is able to quickly recover its direction and position in the road. The vehicle sometimes veers to the side slightly. This could be due to my bad driving when recording. It would be interesting to see if combining different camera views leads to better driving.

Track two has some strong shadows and it is interesting to see that the model worked well without further data augmentation. Of course, one could have augmented the training data by adding shadows in some random way.

Using the same architecture and training methods, I also ran two experiments training on recordings from one track and driving on the other. In neither of the settings did the model manage to drive more than 1/4 of the track. One cause of this could simply be that the steering angles are very different on the two tracks. It would be very interesting to observe how the model behaves on a third track.

