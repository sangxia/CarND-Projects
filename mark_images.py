import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import marklane

def process_image(f, dist_info, pp_mtx, pp_mtx_inv, \
        window_width=150, window_height=50, margin=150):
    img = cv2.imread(f)
    img_ud = marklane.undistort(img, dist_info)
    binary = marklane.binary_lane_threshold(img_ud)
    warped = marklane.warp_img(binary, pp_mtx)
    centroids = marklane.find_window_centroids(warped, window_width, window_height, margin)
    lx, ly, rx, ry, flx, fly, frx, fry = \
            marklane.fit_lane_centroids(warped, centroids, window_width, window_height)
    coeff, score = marklane.fit_lane_poly(warped, window_width, window_height, \
            flx, fly, frx, fry, np.zeros(4), 0)
    warped_fit = marklane.fit_warped(\
            warped, centroids, window_width, window_height, \
            lx, ly, rx, ry, coeff)
    result = marklane.draw_lanes(img_ud, coeff, pp_mtx_inv)
    return img_ud, \
            np.dstack([binary,binary,binary]), \
            np.dstack([warped,warped,warped]), \
            warped_fit, result[0]

with open('distort_calibration.pickle', 'rb') as f:
    dist_info = pickle.load(f)

pp_mtx, pp_mtx_inv = marklane.get_perspective_matrix()

#path_in = 'video_images/'
#path_out = 'output_video_images/'
#images = ['video_images/harder_challenge_video_7.jpg']
path_in = 'test_images/'
path_out = 'output_images/'
images = glob.glob(path_in + '*')
for f in images:
    print('processing ', f)
    ud, bn, wp, wf, out = process_image(f, dist_info, pp_mtx, pp_mtx_inv)
    for prefix, img in zip(\
            ['ud','bn','wp','wf','final'], \
            [ud, bn, wp, wf, out]):
        print(prefix)
        fout = path_out + prefix + '_' + f[len(path_in):]
        cv2.imwrite(fout, img)

