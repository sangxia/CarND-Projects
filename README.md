# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

In this project, your goal is to write a software pipeline to detect vehicles in a video.

## The Project

The goals / steps of this project are the following:

* Extract features such as Histogram of Oriented Gradients (HOG) and binned color histogram to train a classifier using a labeled training set of images
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Results

* writeup.md summarizing the project
* utils.py containing code for most of the pipeline
* mark_images.py and mark_videos.py containing code that detect vehicles for images and videos
* output_project_video.mp4 is the final video output for the project
* prepare_features_cv2.py, validation.py, validate_on_crop.py containing code that extract feature and train classifier on 64x64 vehicle and non-vehicle images
