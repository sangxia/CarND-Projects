import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import delayed, Parallel
from itertools import product

def run(feature_list, feature_set, C, gamma, \
        train_features, train_labels, \
        test_features, test_labels):
    feature_col = []
    col = 0
    for fname, fsize in feature_list:
        if fname in feature_set:
            feature_col += list(range(col,col+fsize))
        col += fsize
    trainX = train_features[:, feature_col]
    testX = test_features[:, feature_col]
    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)
    clf = SVC(C=C, gamma=gamma/len(feature_col), probability=True)
    clf.fit(trainX, train_labels)
    pred = clf.predict(testX)
    proba = clf.predict_proba(testX)
    acc = [np.mean((pred==1) & (test_labels==1))/np.mean(test_labels==1), \
            np.mean((pred==0) & (test_labels==1))/np.mean(test_labels==1), \
            np.mean((pred==1) & (test_labels==0))/np.mean(test_labels==0), \
            np.mean((pred==0) & (test_labels==0))/np.mean(test_labels==0), \
            np.mean(pred==test_labels)]
    print('gamma {0:.2f} C {1:.2f} features {2} {3}'.format(\
            gamma, C, feature_set, \
            ' '.join('{:.4f}'.format(x) for x in acc)))
    names = sorted([str(x) for x in range(test_labels.shape[0])])
    for i in range(test_labels.shape[0]):
        print(names[i], test_labels[i], pred[i], proba[i])

feature_list = [\
        ('h_histogram', 15), \
        ('y_hog', 1764), \
        ('l_hog', 1764), \
        ('s_hls_hog', 1764), \
        ('s_hsv_hog', 1764), \
        ('v_hog', 1764), \
        ('b_hog', 1764), \
        ('g_hog', 1764), \
        ('r_hog', 1764)]

print('loading data')
vehicle = np.load('vehicle_feature.npy')
nonvehicle = np.load('nonvehicle_feature.npy')
# consumes about about 6G ram here
print('vehicle', vehicle.shape)
print('nonvehicle', nonvehicle.shape)
train_features = np.concatenate([vehicle, nonvehicle])
train_labels = np.concatenate([np.ones(vehicle.shape[0]), \
        np.zeros(nonvehicle.shape[0])])

print('loading cropped test images')
test_features = np.load('cropped_test_images/cropped_test_data.npy')
with open('cropped_test_images/labels.pickle','rb') as f:
    test_labels = pickle.load(f)

# rearrange because I sorted the files wrong
test_labels = test_labels[\
        [0,1] + list(range(10,20)) + [2] + list(range(20,30)) + \
        [3] + list(range(30,40)) + [4, 40] + list(range(5,10))]

print('training')
run(feature_list, set(['h_histogram', 'l_hog']), 2., 0.9, \
        train_features, train_labels, test_features, test_labels)

