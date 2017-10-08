import glob
import cv2
import numpy as np
from utils import get_features
import pickle

images = sorted(glob.glob('cropped_test_images/*.jpg'))
features = []
for fname in images:
    img = cv2.imread(fname)
    features.append(get_features(img))

features = np.stack(features)
print(features.shape)
np.save('cropped_test_data.npy', features)

