import numpy as np
from skimage.feature import hog
from scipy.ndimage.filters import convolve
import cv2
from itertools import product
from scipy.ndimage.measurements import label
from joblib import delayed, Parallel

def channel_hog_cv2(img, hd):
    res = hd.compute(img, locations=[(1,1)])
    return res.reshape(-1)

#def channel_hog(img, n_orient=9, pix_per_cell=8, cell_per_block=2, \
#                     transform_sqrt = False, \
#                     vis=False, feature_vec=True, hd=None):
#    # in skimage 0.12.3 L2 block-normalization is applied automatically
#    if hd is not None:
#        return hd.compute(img).reshape(-1)
#    if vis:
#        features, hog_image = hog(img, orientations=n_orient, \
#                                  pixels_per_cell=(pix_per_cell, pix_per_cell), \
#                                  cells_per_block=(cell_per_block, cell_per_block), \
#                                  transform_sqrt=transform_sqrt, \
#                                  visualise=vis, feature_vector=feature_vec)
#        return features, hog_image
#    else:      
#        features = hog(img, orientations=n_orient, \
#                       pixels_per_cell=(pix_per_cell, pix_per_cell), \
#                       cells_per_block=(cell_per_block, cell_per_block), \
#                       transform_sqrt=transform_sqrt, \
#                       visualise=vis, feature_vector=feature_vec)
#        return features

def channel_hist(img, nbins=32, bins_range=(0, 256)):
    return np.histogram(img, bins=nbins, range=bins_range)[0]

def get_features_cv2(img, hd):
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
            features.append(channel_hog_cv2(ch,hd))
        else:
            features.append(channel_hog_cv2(ch[:,:,0],hd))
    return np.concatenate(features)

def perform_feature_extraction_cv2(hls, img_hog, r, c, base_r, base_c, coord_scale):
    """ the location parameters are recorded for inferring boxes """
    cell_hist = channel_hist(hls[r*8:r*8+64, c*8:c*8+64, 0], nbins=15, bins_range=(0,180))
    cell_feat = img_hog[r,c,:]
    return np.concatenate([[base_r, base_c, r, c, coord_scale], cell_hist, cell_feat])

def perform_hog_cv2(l):
    hd = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    return hd.compute(l).reshape(l.shape[0]//8-7, l.shape[1]//8-7, 1764)

def crop_and_hls(img, x):
    return cv2.cvtColor(cv2.resize(\
            img[x[0]:x[2],x[1]:x[3],:], (x[5],x[4])), cv2.COLOR_BGR2HLS)

def detect_vehicles_from_crops(img, crop_list, scaler, clf):
    n_jobs = 8
    images = [crop_and_hls(img, x) for x in crop_list]
    hogs = [perform_hog_cv2(img[:,:,1]) for img in images]
    # matrix to store extracted features
    features = []
    ncells_per_window = 8
    paramslist = []
    for i,cp in enumerate(crop_list):
        max_row = ((images[i].shape[0]-1)//8)-ncells_per_window
        max_col = ((images[i].shape[1]-1)//8)-ncells_per_window
        cell_stride = cp[-1]
        paramslist += [
                (i, r, c, cp[0], cp[1], (cp[2]-cp[0])/cp[4]) \
                for r,c in product(\
                range(1,max_row,cell_stride), \
                range(1,max_col,cell_stride))]
    features = Parallel(n_jobs=n_jobs)(\
            delayed(perform_feature_extraction_cv2)(\
            images[param[0]], hogs[param[0]], param[1], param[2], param[3], param[4], param[5]) \
            for param in paramslist)
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
    # then scaling factor
    # final number is cell_stride
    crop_list = [\
            #(464,1280, 208,2560, 4), \
            (360,0, 496,1280, 136,1280, 1., 2), \
            (360,0, 560,1280, 150,960, 4/3, 2), \
            (360,0, 592,1280, 145,800, 1.6, 2), \
            (360,0, 656,1280, 148,640, 2., 1), \
            (360,0, 656,1280, 111,480, 8/3, 1), \
            #(656,1280, 74,320, 1), \
            ]
    # size is (32,) 64, 85.33, 102.4, 128, 170.66, 256
    return detect_vehicles_from_crops(img, crop_list, scaler, clf)

def box_intersection(box1, box2):
    """
    box1 and box2 are 4-tuples
    each has coordinates of top left followed by bottom right
    returns the intersection if it exists, None if intersection
    is empty
    """
    box = (max(box1[0], box2[0]), max(box1[1], box2[1]), \
            min(box1[2], box2[2]), min(box1[3], box2[3]))
    if (box[0]<box[2]) and (box[1]<box[3]):
        return box
    else:
        return None

def expand_box(box, expansion):
    return (box[0]-expansion, box[1]-expansion, box[2]+expansion, box[3]+expansion)

def box_union(box1, box2):
    """
    box1 and box2 are 4-tuples
    each has coordinates of top left followed by bottom right
    returns the intersection if it exists, None if intersection
    is empty
    """
    box = (min(box1[0], box2[0]), min(box1[1], box2[1]), \
            max(box1[2], box2[2]), max(box1[3], box2[3]))
    return box

def box_area(box):
    if box is None:
        return 0
    else:
        return (box[3]-box[1])*(box[2]-box[0])

def iou(b1, b2):
    return box_area(box_intersection(b1, b2))/box_area(box_union(b1, b2))

def iomin(b1, b2):
    return box_area(box_intersection(b1, b2))/min(box_area(b1), box_area(b2))

def average_box(w1, b1, w2, b2):
    return tuple(int(w1*v1+w2*v2) for v1,v2 in zip(b1,b2))

def detect_vehicles_in_boxes_parallel(img, boxes, scaler, clf):
    """
    boxes contains a list of 4-tuples describing the top left
    and bottom right of the regions in the images
    detection will only be performed on those regions

    for video detection, boxes would be detections from previous
    frames (slightly expanded) as well as regions on the left and
    right side of the frame
    """
    # each row_range tuple correspond to one scaling setting
    # first two are the top and bottom row number
    # third is the scaling factor
    # last is cell_stride
    row_range = [\
            (360,496,1.,2), \
            (360,560,4/3,2), \
            (360,592,1.6,2), \
            (360,656,2.,1), \
            (360,656,8/3,1)]
    crop_list = []
    for box,rg in product(boxes, row_range):
        crop_box = box_intersection(box, (rg[0],0,rg[1],img.shape[1]))
        if crop_box:
            target_size = (int((crop_box[2]-crop_box[0])/rg[2]), \
                    int((crop_box[3]-crop_box[1])/rg[2]))
            if min(target_size)>=64:
                crop_list.append(crop_box+target_size+(rg[2],rg[3]))
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

def get_bboxes(hm, lbls, size_filter=True):
    result = []
    for i in range(lbls):
        bboxes = []
        lst = np.where(hm==(i+1))
        box = (np.min(lst[0]), np.min(lst[1]), np.max(lst[0]), np.max(lst[1]))
        if size_filter and ((box[2]<=box[0]) or (box[3]<=box[1]) or \
                ((box[3]-box[1])/(box[2]-box[0]) > 2) or \
                ((box[3]-box[1])/(box[2]-box[0]) < 0.5)):
            continue
        result.append((\
                np.min(lst[0]), np.min(lst[1]), \
                np.max(lst[0]), np.max(lst[1])))
    return result

def draw_boxes(img, bboxes, color=(0, 255, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, tuple(bbox[1::-1]), tuple(bbox[3:1:-1]), color, thick)
    return imcopy

