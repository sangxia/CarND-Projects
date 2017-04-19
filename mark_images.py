import pickle
import cv2
import numpy as np
import utils
import glob
import joblib

model_fname = 'model_fullset'
scaler, clf = joblib.load(model_fname + '_scaler.pickle'), \
        joblib.load(model_fname + '_svc.pickle')

file_list = sorted(glob.glob('test_images/*.jpg'))
output_dir = 'output_images/'
for f in file_list:
    fname = f[f.rindex('/')+1:]
    print('Processing', fname)
    img = cv2.imread(f)
    boxes = utils.detect_vehicles_parallel(img, scaler, clf)
    cv2.imwrite(output_dir + 'raw_boxes_' + fname, \
            utils.draw_boxes(img, boxes))
    heatmap = utils.get_heatmap(img.shape, boxes, 1)
    cv2.imwrite(output_dir + 'hm_no_thresh_' + fname, \
            (heatmap * (255//np.max(heatmap))).astype(np.uint8))
    heatmap = utils.get_heatmap(img.shape, boxes, 3)
    cv2.imwrite(output_dir + 'hm_thresh_' + fname, \
            (255*heatmap).astype(np.uint8))
    hmlbl, lbls = utils.get_labels(heatmap)
    if lbls:
        cv2.imwrite(output_dir + 'labeled_hm_' + fname, \
                (hmlbl*(255//lbls)).astype(np.uint8))
    else:
        cv2.imwrite(output_dir + 'labeled_hm_' + fname, \
                hmlbl.astype(np.uint8))
    bboxes = utils.get_bboxes(hmlbl, lbls)
    img_final = utils.draw_boxes(img, bboxes)
    cv2.imwrite(output_dir + 'final_' + fname, img_final)

