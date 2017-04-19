import numpy as np
from skimage.feature import hog
from scipy.ndimage.filters import convolve
import cv2
from itertools import product
from scipy.ndimage.measurements import label
from joblib import delayed, Parallel

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

#def get_features_for_detection(img):
#    """
#    img should be cropped and scaled 
#    both dimension of the image should be multiples of 8
#    the assumption is that crops of img of size 64x64 will
#    be fed into a classifier
#
#    returns hog feature, and hue channel histogram with 
#    15 bins for 8x8 cells
#    """
#    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#    hog_feature = channel_hog(hls[:,:,1], feature_vec=False)
#    h_channel = (hls[:,:,0]/12).astype(np.int)
#    # do one-hot encoding for the histogram
#    h_hist = (np.arange(15) == h_channel[:,:,None]).astype(int)
#    weight = np.dstack(\
#            [np.zeros((8,8)) for _ in range(7)] + \
#            [np.ones((8,8))] + \
#            [np.zeros((8,8)) for _ in range(7)])
#    h_hist = convolve(h_hist, weight, mode='constant')[3::8,3::8,:]
#    return hog_feature, h_hist

#def detect_vehicles_single_scale(img, scaler, clf, cell_stride=2, \
#        coord_scale=1.0, base_r=0, base_c=0, channeled=False):
#    """
#    img should be cropped and scaled 
#    both dimension of the image should be multiples of 8
#    the algorithm looks for windows of size 64x64 pixels
#
#    results is a list of tuples containing the upper left 
#    corner of the vehicle windows
#    """
#    results = []
#    if not channeled:
#        channels = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#    else:
#        channels = img
#    ncells_per_window = 8
#    max_row = (img.shape[0]//8)-ncells_per_window
#    max_col = (img.shape[1]//8)-ncells_per_window
#    for r,c in product(range(0,max_row,cell_stride), range(0,max_col,cell_stride)):
#        cell_hist = channel_hist(channels[r*8:r*8+64, c*8:c*8+64, 0], nbins=15, bins_range=(0,180))
#        cell_feat = channel_hog(channels[r*8:r*8+64, c*8:c*8+64, 1])
#        feat = np.concatenate([cell_hist, cell_feat]).reshape(1,-1)
#        scaled_feat = scaler.transform(feat)
#        pred = clf.predict(scaled_feat)
#        if pred:
#            results.append((\
#                    int(base_r+coord_scale*r*8),int(base_c+coord_scale*c*8), \
#                    int(base_r+coord_scale*r*8+64*coord_scale), int(base_c+coord_scale*c*8+64*coord_scale)))
#
#    return results

def perform_feature_extraction(hls, r, c, base_r, base_c, coord_scale):
    """ the location parameters are recorded for inferring boxes """
    cell_hist = channel_hist(hls[r*8:r*8+64, c*8:c*8+64, 0], nbins=15, bins_range=(0,180))
    cell_feat = channel_hog(hls[r*8:r*8+64, c*8:c*8+64, 1])
    return np.concatenate([[base_r, base_c, r, c, coord_scale], cell_hist, cell_feat])

def crop_and_hls(img, x):
    return cv2.cvtColor(cv2.resize(\
            img[x[0]:x[2],x[1]:x[3],:], (x[5],x[4])), cv2.COLOR_BGR2HLS)

def detect_vehicles_from_crops(img, crop_list, scaler, clf):
    n_jobs = 8
    images = [crop_and_hls(img, x) for x in crop_list]
    # matrix to store extracted features
    features = []
    ncells_per_window = 8
    for cp,image in zip(crop_list, images):
        max_row = (image.shape[0]//8)-ncells_per_window
        max_col = (image.shape[1]//8)-ncells_per_window
        cell_stride = cp[-1]
        features += Parallel(n_jobs=n_jobs)(\
                delayed(perform_feature_extraction)(\
                image, r, c, cp[0], cp[1], (cp[2]-cp[0])/cp[4]) \
                for r,c in product(\
                range(0,max_row,cell_stride), \
                range(0,max_col,cell_stride)))
    features = np.stack(features)
    features[:,5:] = scaler.transform(features[:,5:])
    step = features.shape[0] // n_jobs
    results = Parallel(n_jobs=n_jobs)(\
            delayed(clf.predict)(features[start:start+step,5:]) \
            for start in range(0,features.shape[0],step))
    results = np.concatenate([features[:,:5], \
            np.concatenate(results)[:,None]], axis=1)
    results = results[results[:,-1]==1]
    results = np.stack([\
            results[:,0]+results[:,2]*results[:,4]*8, \
            results[:,1]+results[:,3]*results[:,4]*8, \
            results[:,0]+results[:,2]*results[:,4]*8+results[:,4]*64, \
            results[:,1]+results[:,3]*results[:,4]*8+results[:,4]*64], \
            axis=1).astype(np.int)
    return list(results)

def detect_vehicles_parallel(img, scaler, clf):
    # crops always has top left (360,0)
    # the first two numbers are coordinate of top left
    # followed by coordinates of bottom right
    # the next two are the dst size
    # final number is cell_stride
    crop_list = [\
            #(464,1280, 208,2560, 4), \
            (360,0, 496,1280, 136,1280, 2), \
            (360,0, 560,1280, 150,960, 2), \
            (360,0, 592,1280, 145,800, 2), \
            (360,0, 656,1280, 148,640, 1), \
            (360,0, 656,1280, 111,480, 1), \
            #(656,1280, 74,320, 1), \
            ]
    # size is (32,) 64, 85.33, 102.4, 128, 170.66, 256
    return detect_vehicles_from_crops(img, crop_list, scaler, clf)

def detect_vehicles_in_boxes_parallel(img, boxes, scaler, clf):
    # each tuple correspond to one scaling setting
    # first two are the top and bottom row number
    # third is the scaling factor
    # last is cell_stride
    row_range = [\
            (360,496,1.,2), \
            (360,560,4/3,2), \
            (360,592,1.6,2), \
            (360,656,2.,1), \
            (360,656,8/3,1)]


    return detect_vehicles_from_crops(img, crop_list, scaler, clf)

def get_heatmap(shape, boxes, thresh):
    if len(shape) == 2:
        heatmap = np.zeros(shape)
    else:
        heatmap = np.zeros(shape[:2])
    for b in boxes:
        heatmap[b[0]:b[2],b[1]:b[3]] += 1
    heatmap = (heatmap>=thresh)
    return heatmap

def get_labels(heatmap):
    return label(heatmap)

def get_bboxes(hm, lbls):
    result = []
    for i in range(lbls):
        bboxes = []
        lst = np.where(hm==(i+1))
        result.append((\
                np.min(lst[0]), np.min(lst[1]), \
                np.max(lst[0]), np.max(lst[1])))
    return result

def draw_boxes(img, bboxes, color=(0, 255, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, tuple(bbox[1::-1]), tuple(bbox[3:1:-1]), color, thick)
    return imcopy

