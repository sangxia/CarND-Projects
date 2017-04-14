import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import marklane
from moviepy.editor import VideoFileClip
from moviepy.config import change_settings
change_settings({'FFMPEG_BINARY': '/usr/bin/ffmpeg'})

logs = {\
        'null_count': 0, \
        'coeff': np.zeros(4), \
        'score': 0.}

def process_image(img, dist_info, pp_mtx, pp_mtx_inv, logs, \
        window_width=150, window_height=50, margin=80):
    img_ud = marklane.undistort(img, dist_info)
    binary = marklane.binary_lane_threshold(img_ud)
    warped = marklane.warp_img(binary, pp_mtx)
    centroids = marklane.find_window_centroids(warped, window_width, window_height, margin)
    lx, ly, rx, ry, flx, fly, frx, fry = \
            marklane.fit_lane_centroids(warped, centroids, window_width, window_height)
    coeff, score = marklane.fit_lane_poly(warped, window_width, window_height, \
            flx, fly, frx, fry, logs['coeff'], logs['score'])
    logs['coeff'] = coeff
    logs['score'] = score
    result = marklane.draw_lanes(img_ud, coeff, pp_mtx_inv)
    if result[1] is None:
        null_count = logs['null_count']
        cv2.imwrite('error_images/{0}.jpg'.format(null_count), img)
        logs['null_count'] = null_count+1
    return result[0][:,:,::-1]

with open('distort_calibration.pickle', 'rb') as f:
    dist_info = pickle.load(f)

pp_mtx, pp_mtx_inv = marklane.get_perspective_matrix()

fname = 'project_video.mp4'
clip1 = VideoFileClip(fname)
clip2 = clip1.fl_image(lambda img: \
        process_image(img[:,:,::-1], dist_info, pp_mtx, pp_mtx_inv, logs))
clip2.write_videofile('output_'+fname, audio=False)

print(logs['null_count'])

