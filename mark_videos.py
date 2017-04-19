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

def merge_detections(old, new, discount):
    if len(new) == 0:
        results = [(box,score*discount) for box,score in old if score*discount>0.5]
        return results
    results = []
    flags = [0] * len(new)
    for box,score in old:
        candidates = [(bp, utils.iomin(box,bp)) for bp in new]
        best_idx, best_new = max(enumerate(candidates), key=lambda x: x[1][1])
        # if intersection over union score is large
        if best_new[1]>0.6:
            new_box = utils.average_box(discount,box,1-discount,best_new[0])
            if len(results)==0 or \
                    max([utils.iomin(new_box,br[0]) for br in results])<0.2:
                results.append((new_box,score*discount+1))
            flags[best_idx] = 1
        elif score*discount>0.5:
            if len(results)==0 or \
                    max([utils.iomin(box,br[0]) for br in results])<0.2:
                results.append((box, score*discount))
    for i, flag in enumerate(flags):
        if flag==0:
            # only add if it doesn't overlap significantly with existing ones
            if len(results)==0 or \
                    max([utils.iomin(new[i],br[0]) for br in results])<0.2:
                results.append((new[i], 1.))
    # sort high score and large boxes first
    return sorted(results, key=lambda x: (-x[1], -utils.box_area(x[0])))

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
            img, rois, scaler, clf, True)
    heatmap = utils.get_heatmap(img.shape, boxes, 4)
    hmlbl, lbls = utils.get_labels(heatmap)
    bboxes = utils.get_bboxes(hmlbl, lbls)
    print('bboxes',bboxes)
    states['detections'] = merge_detections(\
            states['detections'], \
            sorted(bboxes, key=lambda x: -utils.box_area(x)), \
            states['discount'])
    thresh = (1-states['discount']**states['thresh_frame_count'])/(1-states['discount'])
    return utils.draw_boxes(img, \
            [b[0] for b in states['detections'] if b[1]>thresh])[:,:,::-1]

model_fname = 'model_fullset_cv2'
scaler, clf = joblib.load(model_fname + '_scaler.pickle'), \
        joblib.load(model_fname + '_svc.pickle')

# initial_phase: # of initial frames on which full detection is performed
# count_down: internal state variable, # of frames until the next full detection
# reset: # of frames during which fast detection is performed
# border_region: region of interest on the left and right side of the frame
# discount: score discount for past detections
# thresh_frame_count: min number of detections in consecutive frames for a box
# to be drawn
# detections: a list of previous detections, each a tuple organized as follows
# (4-tuple for box coordinates, detection score)
states = {\
        'initial_phase': 6, \
        'count_down': 0, \
        'reset': 24, \
        'border_region': [(360,960,656,1280),(360,0,656,320)], \
        'discount': 0.9, \
        'thresh_frame_count': 3, \
        'detections': []}

#fname = 'test_video.mp4'
fname = 'project_video.mp4'
clip1 = VideoFileClip(fname)
clip2 = clip1.subclip(25.,45.).fl_image(lambda img: \
        process_image(img[:,:,::-1], scaler, clf, states))
clip2.write_videofile('output_'+fname, audio=False)

