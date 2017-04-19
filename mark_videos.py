import pickle
import cv2
import numpy as np
import utils
import glob
import joblib
from moviepy.editor import VideoFileClip
from moviepy.config import change_settings
import copy
change_settings({'FFMPEG_BINARY': '/usr/bin/ffmpeg'})

def process_image(img, scaler, clf, states):
    if (states['initial_phase']>0) or (states['count_down']==0):
        rois = [(0,0,720,1280)]
        if states['initial_phase']>0:
            states['initial_phase'] -= 1
            if states['initial_phase']==0:
                states['count_down'] = states['reset']
        else:
            states['count_down'] = states['reset']
    else:
        rois = copy.copy(states['border_region'])
        for bb in states['detections']:
            rois.append(utils.expand_box(bb[0], 96))
        states['count_down'] -= 1
    print(len(rois), rois)
    print(states['detections'])
    boxes = utils.detect_vehicles_in_boxes_parallel(\
            img, rois, scaler, clf)
    heatmap = utils.get_heatmap(img.shape, boxes, 4)
    hmlbl, lbls = utils.get_labels(heatmap)
    bboxes = utils.get_bboxes(hmlbl, lbls)
    states['detections'] = [(b,1) for b in bboxes]
    return utils.draw_boxes(img, bboxes)[:,:,::-1]

model_fname = 'model_fullset'
scaler, clf = joblib.load(model_fname + '_scaler.pickle'), \
        joblib.load(model_fname + '_svc.pickle')

# initial_phase: # of initial frames on which full detection is performed
# count_down: internal state variable, # of frames until the next full detection
# reset: # of frames during which fast detection is performed
# border_region: region of interest on the left and right side of the frame
# detections: a list of previous detections, each a tuple organized as follows
# (4-tuple for box coordinates, detection score)
states = {\
        'initial_phase': 6, \
        'count_down': 0, \
        'reset': 8, \
        'border_region': [(360,960,656,1280),(360,0,656,320)], \
        'detections': []}

fname = 'test_video.mp4'
clip1 = VideoFileClip(fname)
clip2 = clip1.fl_image(lambda img: \
        process_image(img[:,:,::-1], scaler, clf, states))
clip2.write_videofile('output_'+fname, audio=False)

