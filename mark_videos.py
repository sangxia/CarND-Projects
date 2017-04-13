import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import marklane
from moviepy.editor import VideoFileClip
from moviepy.config import change_settings
change_settings({'FFMPEG_BINARY': '/usr/bin/ffmpeg'})

null_count = 0

def process_image(img, dist_info, pp_mtx, pp_mtx_inv, \
        window_width=50, window_height=50, margin=50):
    img_ud = marklane.undistort(img, dist_info)
    binary = marklane.binary_lane_threshold(img_ud)
    warped = marklane.warp_img(binary, pp_mtx)
    centroids = marklane.find_window_centroids(warped, window_width, window_height, margin)
    warped_fit = marklane.fit_warped(warped, centroids, window_width, window_height)
    result = marklane.draw_lanes(img_ud, warped, centroids, window_width, window_height, \
            pp_mtx_inv)
    if result[1] is None:
        cv2.imwrite('error_images/{0}.jpg'.format(null_count), result[0])
        null_count += 1
    return result[0][:,:,::-1]

with open('distort_calibration.pickle', 'rb') as f:
    dist_info = pickle.load(f)

pp_mtx, pp_mtx_inv = marklane.get_perspective_matrix()

fname = 'challenge_video.mp4'
clip1 = VideoFileClip(fname)
clip2 = clip1.fl_image(lambda img: \
        process_image(img[:,:,::-1], dist_info, pp_mtx, pp_mtx_inv))
clip2.write_videofile('output_'+fname, audio=False)

print(null_count)
