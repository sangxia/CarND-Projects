import pickle
import cv2
import numpy as np
import utils
import glob
import joblib
from moviepy.editor import VideoFileClip
from moviepy.config import change_settings
change_settings({'FFMPEG_BINARY': '/usr/bin/ffmpeg'})

def process_image(img, scaler, clf, states):
    boxes = utils.detect_vehicles_in_boxes_parallel(\
            img, [(0,0,720,1280)], scaler, clf)
    heatmap = utils.get_heatmap(img.shape, boxes, 4)
    hmlbl, lbls = utils.get_labels(heatmap)
    bboxes = utils.get_bboxes(hmlbl, lbls)
    return utils.draw_boxes(img, bboxes)[:,:,::-1]

model_fname = 'model_fullset'
scaler, clf = joblib.load(model_fname + '_scaler.pickle'), \
        joblib.load(model_fname + '_svc.pickle')

states = {}

fname = 'test_video.mp4'
clip1 = VideoFileClip(fname)
clip2 = clip1.fl_image(lambda img: \
        process_image(img[:,:,::-1], scaler, clf, states))
clip2.write_videofile('output_'+fname, audio=False)

