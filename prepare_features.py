import pickle
import numpy as np
from skimage.feature import hog
import cv2
from utils import get_features

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

