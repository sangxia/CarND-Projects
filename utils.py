import numpy as np
from skimage.feature import hog
from scipy.ndimage.filters import convolve
import cv2
from itertools import product
from scipy.ndimage.measurements import label

def channel_hog(img, n_orient=9, pix_per_cell=8, cell_per_block=2, \
                     transform_sqrt = False, \
                     vis=False, feature_vec=True):
    # in skimage 0.12.3 L2 block-normalization is applied automatically
    if vis:
        features, hog_image = hog(img, orientations=n_orient, \
                                  pixels_per_cell=(pix_per_cell, pix_per_cell), \
                                  cells_per_block=(cell_per_block, cell_per_block), \
                                  transform_sqrt=transform_sqrt, \
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=n_orient, \
                       pixels_per_cell=(pix_per_cell, pix_per_cell), \
                       cells_per_block=(cell_per_block, cell_per_block), \
                       transform_sqrt=transform_sqrt, \
                       visualise=vis, feature_vector=feature_vec)
        return features

def channel_hist(img, nbins=32, bins_range=(0, 256)):
    return np.histogram(img, bins=nbins, range=bins_range)[0]

def get_features(img):
    """
    takes as input img in BGR format, outputs a concatenated
    feature containing hog of the relevant channels and histogram
    of the hue channel
    """
    features = []
    # obtain the relevant channels
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:,:,0]
    h, l, s_hls = np.dsplit(cv2.cvtColor(img, cv2.COLOR_BGR2HLS), 3)
    s_hsv, v = np.dsplit(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1:], 2)
    b, g, r = np.dsplit(img, 3)
    # histogram of hue channel
    features.append(channel_hist(h, nbins=15, bins_range=(0,180)))
    # hog for other channels
    for ch in [y, l, s_hls, s_hsv, v, b, g, r]:
        if len(ch.shape) == 2:
            features.append(channel_hog(ch))
        else:
            features.append(channel_hog(ch[:,:,0]))
    return np.concatenate(features)

def get_features_for_detection(img):
    """
    img should be cropped and scaled 
    both dimension of the image should be multiples of 8
    the assumption is that crops of img of size 64x64 will
    be fed into a classifier

    returns hog feature, and hue channel histogram with 
    15 bins for 8x8 cells
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hog_feature = channel_hog(hls[:,:,1], feature_vec=False)
    h_channel = (hls[:,:,0]/12).astype(np.int)
    # do one-hot encoding for the histogram
    h_hist = (np.arange(15) == h_channel[:,:,None]).astype(int)
    weight = np.dstack(\
            [np.zeros((8,8)) for _ in range(7)] + \
            [np.ones((8,8))] + \
            [np.zeros((8,8)) for _ in range(7)])
    h_hist = convolve(h_hist, weight, mode='constant')[3::8,3::8,:]
    return hog_feature, h_hist

def detect_vehicles_single_scale(img, scaler, clf, coord_scale, channeled=False):
    """
    img should be cropped and scaled 
    both dimension of the image should be multiples of 8
    the algorithm looks for windows of size 64x64 pixels

    results is a list of tuples containing the upper left 
    corner of the vehicle windows
    """
    results = []
    if not channeled:
        channels = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    else:
        channels = img
    ncells_per_window = 8
    cell_stride = 2
    max_row = (img.shape[0]//8)-ncells_per_window
    max_col = (img.shape[1]//8)-ncells_per_window
    for r,c in product(range(0,max_row,cell_stride), range(0,max_col,cell_stride)):
        cell_hist = channel_hist(channels[r*8:r*8+64, c*8:c*8+64, 0], nbins=15, bins_range=(0,180))
        cell_feat = channel_hog(channels[r*8:r*8+64, c*8:c*8+64, 1])
        feat = np.concatenate([cell_hist, cell_feat]).reshape(1,-1)
        scaled_feat = scaler.transform(feat)
        pred = clf.predict(scaled_feat)
        if pred:
            results.append((\
                    int(coord_scale*r*8),int(coord_scale*c*8), \
                    int(coord_scale*r*8+64*coord_scale), int(coord_scale*c*8+64*coord_scale)))

    return results

def get_heatmap(shape, boxes, thresh):
    heatmap = np.zeros(shape)
    for b in boxes:
        heatmap[b[0]:b[2],b[1]:b[3]] += 1
    heatmap = (heatmap>=thresh)
    return heatmap

def get_labels(heatmap):
    return label(heatmap)

