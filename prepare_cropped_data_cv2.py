import glob
import cv2
import numpy as np
from utils import get_features_cv2
import pickle

hd = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)

images = sorted(glob.glob('cropped_test_images/*.jpg'))
features = []
for fname in images:
    img = cv2.imread(fname)
    img_cpy = cv2.resize(img, (66,66))
    features.append(get_features_cv2(img_cpy, hd))

features = np.stack(features)
print(features.shape)
np.save('cropped_test_data_cv2.npy', features)

