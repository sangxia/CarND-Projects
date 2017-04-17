import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import delayed, Parallel
from itertools import product

def run(params, \
        train_features, train_labels, \
        test_features, test_labels):
    feature_set = params['features']
    feature_col = []
    col = 0
    for fname, fsize in params['feature_list']:
        if fname in feature_set:
            feature_col += list(range(col,col+fsize))
        col += fsize
    trainX = train_features[:, feature_col]
    testX = test_features[:, feature_col]
    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)
    clf = SVC(C=params['C'], gamma=params['gamma']/len(feature_col))
    clf.fit(trainX, train_labels)
    pred = clf.predict(testX)
    acc = [np.mean((pred==1) & (test_labels==1))/np.mean(test_labels==1), \
            np.mean((pred==0) & (test_labels==1))/np.mean(test_labels==1), \
            np.mean((pred==1) & (test_labels==0))/np.mean(test_labels==0), \
            np.mean((pred==0) & (test_labels==0))/np.mean(test_labels==0), \
            np.mean(pred==test_labels)]
    print('gamma {0:.2f} C {1:.2f} features {2} {3}'.format(\
            params['gamma'], params['C'], params['features'], \
            ' '.join('{:.4f}'.format(x) for x in acc)))
    return (params,acc)

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

print('loading split')
#with open('validation_split.pickle', 'rb') as f:
#    split_v, split_nv = pickle.load(f)
#train_v, test_v = split_v
#train_nv, test_nv = split_nv
test_v, train_v = list(range(8792-5966)), list(range(8792-5966,8792))
test_nv, train_nv = list(range(8968-3900)), list(range(8968-3900))
#train_v, test_v = np.array(train_v), np.array(test_v)
#train_nv, test_nv = np.array(train_nv), np.array(test_nv)

#train_features = np.concatenate([\
#        vehicle[train_v[:,None],feature_col], \
#        nonvehicle[train_nv[:,None],feature_col]])
#test_features = np.concatenate([\
#        vehicle[test_v[:,None],feature_col], \
#        nonvehicle[test_nv[:,None],feature_col]])

train_features = np.concatenate([vehicle[train_v,:], nonvehicle[train_nv,:]])
test_features = np.concatenate([vehicle[test_v,:], nonvehicle[test_nv,:]])
train_labels = np.concatenate([np.ones(len(train_v)), np.zeros(len(train_nv))])
test_labels = np.concatenate([np.ones(len(test_v)), np.zeros(len(test_nv))])

paramslist = []
for c, gamma,hog_f in product(\
        np.arange(0.8,3.3,0.2), \
        np.arange(0.6,1.6,0.1), \
        [x[0] for x in feature_list if 'hog' in x[0] and (x[0] != 's_hls_hog')]):
    paramslist.append({\
            'feature_list': feature_list, \
            'features': set(['h_histogram', hog_f]), \
            'C': c, \
            'gamma': gamma})

print('training')
results = Parallel(n_jobs=7, max_nbytes=1e6)(\
        delayed(run)(\
        params, \
        train_features, train_labels, \
        test_features, test_labels) \
        for params in paramslist)

with open('results.pickle','wb') as f:
    pickle.dump(results, f)

