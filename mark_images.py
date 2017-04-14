import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import marklane

def process_image(f, dist_info, pp_mtx, pp_mtx_inv, \
        window_width=150, window_height=50, margin=80):
    img = cv2.imread(f)
    img_ud = marklane.undistort(img, dist_info)
    binary = marklane.binary_lane_threshold(img_ud)
    warped = marklane.warp_img(binary, pp_mtx)
    centroids = marklane.find_window_centroids(warped, window_width, window_height, margin)
    lx, ly, rx, ry, flx, fly, frx, fry = \
            marklane.fit_lane_centroids(warped, centroids, window_width, window_height)
    warped_fit = marklane.fit_warped(\
            warped, centroids, window_width, window_height, \
            lx, ly, rx, ry, flx, fly, frx, fry)
    result = marklane.draw_lanes(img_ud, flx, fly, frx, fry, pp_mtx_inv)
    return img_ud, \
            np.dstack([binary,binary,binary]), \
            np.dstack([warped,warped,warped]), \
            warped_fit, result[0]

with open('distort_calibration.pickle', 'rb') as f:
    dist_info = pickle.load(f)

pp_mtx, pp_mtx_inv = marklane.get_perspective_matrix()

images = glob.glob('test_images/*')
for f in images:
    print('processing ', f)
    ud, bn, wp, wf, out = process_image(f, dist_info, pp_mtx, pp_mtx_inv)
    for prefix, img in zip(\
            ['ud','bn','wp','wf','final'], \
            [ud, bn, wp, wf, out]):
        print(prefix)
        fout = 'output_images/' + prefix + '_' + f[len('test_images/'):]
        cv2.imwrite(fout, img)

