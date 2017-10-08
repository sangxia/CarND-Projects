import pickle
import numpy as np
from skimage.feature import hog
import cv2
from utils import get_features
import time

use_cv2_hog = False
if use_cv2_hog:
    # win, block, block stride, cell, bins
    hd = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
else:
    hd = None
 
print('Loading data')
time_start = time.time()
with open('dataset-data.pickle', 'rb') as f:
    vehicle_data, nonvehicle_data = pickle.load(f)
time_end = time.time()
print('load time {:.3f}'.format(time_end-time_start))

for data, fname in zip(\
        [vehicle_data, nonvehicle_data], \
        ['vehicle_feature', 'nonvehicle_feature']):
    time_start = time.time()
    features = []
    names = sorted(list(data.keys()))
    print(fname, names)
    for p in names:
        print(p)
        for img in data[p]:
            features.append(get_features(img, hd=hd))
    features = np.stack(features)
    print(features.shape)
    print(features[:5,100:110])
    np.save(fname+'_cv2.npy.tmp', features)
    time_end = time.time()
    print('computation time {:.3f}'.format(time_end-time_start))
