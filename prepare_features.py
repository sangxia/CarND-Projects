import pickle
import numpy as np
from skimage.feature import hog
import cv2

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
    # Compute the histogram of the color channels separately
    return np.histogram(img, bins=nbins, range=bins_range)[0]

def get_features(img):
    # img is in BGR format
    features = []
    # obtain the relevant channels
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:,:,0]
    h, l, s_hls = np.dsplit(cv2.cvtColor(img, cv2.COLOR_BGR2HLS), 3)
    s_hsv, v = np.dsplit(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1:], 2)
    b, g, r = np.dsplit(img, 3)
    # histogram for hue channel
    features.append(channel_hist(h, nbins=15, bins_range=(0,180)))
    # hog for the other channels
    for ch in [y, l, s_hls, s_hsv, v, b, g, r]:
        if len(ch.shape) == 2:
            features.append(channel_hog(ch))
        else:
            features.append(channel_hog(ch[:,:,0]))
    return np.concatenate(features)

print('Loading data')
with open('dataset-data.pickle', 'rb') as f:
    vehicle_data, nonvehicle_data = pickle.load(f)

for data, fname in zip(\
        [vehicle_data, nonvehicle_data], \
        ['vehicle_feature', 'nonvehicle_feature']):
    features = []
    names = sorted(list(data.keys()))
    print(fname, names)
    for p in names:
        print(p)
        for img in data[p]:
            features.append(get_features(img))
    features = np.stack(features)
    print(features.shape)
    np.save(fname+'.npy', features)

