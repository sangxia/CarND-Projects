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
    clf = SVC(C=params['C'], \
            gamma=params['gamma']/len(feature_col), \
            probability=False) # probability=True seems significantly slower
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
hogset = set(['y_hog', 'l_hog', 's_hsv_hog', 'b_hog', 'g_hog'])

print('loading data')
vehicle = np.load('vehicle_feature.npy')
nonvehicle = np.load('nonvehicle_feature.npy')
# consumes about about 6G ram here
print('vehicle', vehicle.shape)
print('nonvehicle', nonvehicle.shape)

print('loading split')
split_ratio = 0.7
train_v = list(range(int(834*split_ratio))) + \
        list(range(834,834+int(909*split_ratio))) + \
        list(range(834+909,834+909+int(419*split_ratio))) + \
        list(range(834+909+419,834+909+419+int(664*split_ratio))) + \
        list(range(834+909+419+664,834+909+419+664+int(5966*split_ratio)))
train_v_set = set(train_v)
test_v = [x for x in range(8792) if x not in train_v_set]
train_nv = list(range(int(5068*split_ratio))) + \
        list(range(5068,5068+int(3900*split_ratio)))
train_nv_set = set(train_nv)
test_nv = [x for x in range(8968) if x not in train_nv_set]

train_features = np.concatenate([vehicle[train_v,:], nonvehicle[train_nv,:]])
test_features = np.concatenate([vehicle[test_v,:], nonvehicle[test_nv,:]])
train_labels = np.concatenate([np.ones(len(train_v)), np.zeros(len(train_nv))])
test_labels = np.concatenate([np.ones(len(test_v)), np.zeros(len(test_nv))])

paramslist = []
for c, gamma in product(\
        np.arange(1.8,2.3,0.1), \
        np.arange(0.6,1.3,0.1)):
    paramslist.append({\
            'feature_list': feature_list, \
            'features': set(['l_hog']), \
            'C': c, \
            'gamma': gamma})

#for c, gamma,hog_f in product(\
#        np.arange(1.6,2.5,0.2), \
#        np.arange(0.5,1.7,0.1), \
#        hogset):
#    paramslist.append({\
#            'feature_list': feature_list, \
#            'features': set(['h_histogram', hog_f]), \
#            'C': c, \
#            'gamma': gamma})


print('training, total', len(paramslist))
results = Parallel(n_jobs=7, max_nbytes=1e6)(\
        delayed(run)(\
        params, \
        train_features, train_labels, \
        test_features, test_labels) \
        for params in paramslist)

with open('results.pickle','wb') as f:
    pickle.dump(results, f)

