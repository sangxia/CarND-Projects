# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video.

## The Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Results

* writeup.md summarizing the project
* marklane.py containing the code for identifying lanes
* mark_images.py and mark_videos.py containing code that marks lanes in images and videos
* output_project_video.mp4, output_challenge_video.mp4 and output_harder_challenge_video.mp4 containing videos with marked lane area
* output_images containing images of various stages of the pipeline when processing images from test_images
