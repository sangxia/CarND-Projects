import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import delayed, Parallel, dump
from itertools import product

def run(feature_list, feature_set, C, gamma, \
        train_features, train_labels, \
        test_features, test_labels, \
        model_filename=None):
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
    clf = SVC(C=C, gamma=gamma/len(feature_col), probability=False)
    clf.fit(trainX, train_labels)
    pred = clf.predict(testX)
#    proba = clf.predict_proba(testX)
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
        print(names[i], test_labels[i], pred[i])
#        print(names[i], test_labels[i], pred[i], proba[i])
    if model_filename:
        dump(clf, model_filename + '_svc.pickle')
        dump(scaler, model_filename + '_scaler.pickle')

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
print('vehicle', vehicle.shape)
print('nonvehicle', nonvehicle.shape)
train_v = list(range(8792))
train_nv = list(range(8968))
#train_v = list(range(8792-5966,8792))
#train_nv = list(range(5068))
train_features = np.concatenate([vehicle[train_v,:], nonvehicle[train_nv,:]])
train_labels = np.concatenate([np.ones(len(train_v)), \
        np.zeros(len(train_nv))])

print('loading cropped test images')
test_features = np.load('cropped_test_images/cropped_test_data.npy')
with open('cropped_test_images/labels.pickle','rb') as f:
    test_labels = pickle.load(f)

# rearrange because I sorted the files wrong
test_labels = test_labels[\
        [0,1] + list(range(10,20)) + [2] + list(range(20,30)) + \
        [3] + list(range(30,40)) + [4, 40] + list(range(5,10))]

print('training')
run(feature_list, set(['l_hog']), 2.1, 1.1, \
        train_features, train_labels, test_features, test_labels, \
        model_filename='model_fullset_lhog')

