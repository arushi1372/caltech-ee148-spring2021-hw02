import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    intersection = 0
    tlr1, tlc1, brr1, brc1 = box_1[0], box_1[1], box_1[2], box_1[3]
    tlr2, tlc2, brr2, brc2 = box_2[0], box_2[1], box_2[2], box_2[3]
    dx = min(brr1, brr2) - max(tlr1, tlr2)
    dy = min(brc1, brc1) - max(tlc1, tlc2)
    if (dx>=0) and (dy>=0):
        intersection = dx * dy

    area1 = (brc1 - tlc1) * (brr1 - tlr1)
    area2 = (brc2 - tlc2) * (brr2 - tlr2)
    union = area1 + area2 - intersection 
    iou = intersection / union
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        for i in range(len(gt)):
            not_found = True
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if iou >= iou_thr and pred[j][4] >= conf_thr:
                    TP += 1
                    not_found = False
                    break
                elif pred[j][4] >= conf_thr:
                    FP += 1
                    not_found = False
                    break
            if not_found:
                FN += 1

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = 'hw02_preds'
gts_path = 'hw02_annotations'

# load splits:
split_path = 'hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

def compute_PR(iou, preds, gts):
    lst = []
    for fname in preds:
        if preds[fname] != []:
            for pred in preds[fname]:
                lst.append(pred[4])
    confidence_thrs = np.sort(np.array(lst,dtype=float)) # using (ascending) list of confidence scores as thresholds
    tp = np.zeros(len(confidence_thrs))
    fp = np.zeros(len(confidence_thrs))
    fn = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp[i], fp[i], fn[i] = compute_counts(preds, gts, iou_thr=iou, conf_thr=conf_thr)

    # Plot training set PR curves
    recall = np.zeros(len(confidence_thrs))
    precision = np.zeros(len(confidence_thrs))
    for i, elem in enumerate(tp):
        precision[i] = tp[i]/(tp[i] + fp[i])
        recall[i] = tp[i]/(tp[i] + fn[i])
    
    return recall, precision

recall, precision = compute_PR(0.5, preds_train, gts_train)
recall_l, precision_l = compute_PR(0.25, preds_train, gts_train)
recall_m, precision_m = compute_PR(0.75, preds_train, gts_train)

plt.plot(recall, precision, color='black', marker='o')
plt.plot(recall_l, precision_l, color='blue', marker='o')
plt.plot(recall_m, precision_m, color='green', marker='o')
plt.legend(["IOU 0.5", "IOU 0.25", "IOU 0.75"])
plt.title("PR Curves Training")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

if done_tweaking:
    print('Code for plotting test set PR curves.')

    recall, precision = compute_PR(0.5, preds_test, gts_test)
    recall_l, precision_l = compute_PR(0.25, preds_test, gts_test)
    recall_m, precision_m = compute_PR(0.75, preds_test, gts_test)

    plt.figure()
    plt.plot(recall, precision, color='black', marker='o')
    plt.plot(recall_l, precision_l, color='blue', marker='o')
    plt.plot(recall_m, precision_m, color='green', marker='o')
    plt.legend(["IOU 0.5", "IOU 0.25", "IOU 0.75"])
    plt.title("PR Curves Testing")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


