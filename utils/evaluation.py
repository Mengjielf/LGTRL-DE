"""
    This file contains evaluation methods that take in a set of predicted labels
        and a set of ground truth labels and calculate precision, recall, accuracy, f1, and metrics @k
        https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py
"""
import tensorflow as tf
from collections import defaultdict
import csv
import json
import numpy as np
import os
import sys
from sklearn.metrics import precision_recall_curve,roc_curve, auc,roc_auc_score
import keras.backend as K


def evaluate(y_pred,y_test):
    y_pred = np.array(y_pred)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_test)):
        if y_pred[i] >= 0.5:  # 预测为1
            if y_test[i] == 1:  # 确实为1
                tp += 1
            else:  # 应该是0
                fp += 1
        else:  # 预测为0
            if y_test[i] == 1:  # 应该是1
                fn += 1
            else:  # 确实是0
                tn += 1
    print('TP:', tp, 'TN:', tn, 'FP:', fp, 'FN:', fn)
    precision = tp / (tp + fp)
    print('precision:',precision)
    recall = tp / (tp + fn)
    print('recall:',recall)
    f1_score = (2 * precision * recall) / (precision + recall)
    print('f1_score:', f1_score)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    Auc = auc(fpr, tpr)
    print('Auc:', Auc)
    if len(y_pred.shape) == 1:
        y_pred = np.stack([1 - y_pred, y_pred]).transpose((1, 0))

    (precisions, recalls, thresholds) = precision_recall_curve(y_test, y_pred)
    Auprc = auc(recalls, precisions)
    print('Auprc:', Auprc)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    print('minpse:', minpse)

    return recall, precision, f1_score, Auc, Auprc, minpse

# def auroc(y_true, y_pred):
#     return tf.compat.v1.py_func(roc_auc_score, (y_true, y_pred), tf.double)

# def AUC(y_true, y_pred):
#     auc = tf.compat.v1.metrics.auc(y_true, y_pred)[1]
#     tf.compat.v1.Session().run(tf.compat.v1.local_variables_initializer())
#
#     return auc

def auroc(true, pred):

    # We want strictly 1D arrays - cannot have (batch, 1), for instance
    true = K.flatten(true)
    pred = K.flatten(pred)

    # total number of elements in this batch
    totalCount = K.shape(true)[0]

    # sorting the prediction values in descending order
    values, indices = tf.nn.top_k(pred, k=totalCount)
    # sorting the ground truth values based on the predictions above
    sortedTrue = K.gather(true, indices)

    # getting the ground negative elements (already sorted above)
    negatives = 1 - sortedTrue

    # the true positive count per threshold
    TPCurve = K.cumsum(sortedTrue)

    # area under the curve
    auc = K.sum(TPCurve * negatives)

    # normalizing the result between 0 and 1
    totalCount = K.cast(totalCount, K.floatx())
    positiveCount = K.sum(true)
    negativeCount = totalCount - positiveCount
    totalArea = positiveCount * negativeCount
    return auc / totalArea

def Calculate_Mean_Std_Var(recall, precision, f1_score, Auc, Auprc, minpse):
    print("Calculate recall's std:")
    print("均值:", np.mean(recall))
    print("方差:", np.var(recall))
    print("标准差:", np.std(recall, ddof=1))

    print("Calculate precision's std:")
    print("均值:", np.mean(precision))
    print("方差:", np.var(precision))
    print("标准差:", np.std(precision, ddof=1))

    print("Calculate f1_score's std:")
    print("均值:", np.mean(f1_score))
    print("方差:", np.var(f1_score))
    print("标准差:", np.std(f1_score, ddof=1))

    print("Calculate Auc's std:")
    print("均值:", np.mean(Auc))
    print("方差:", np.var(Auc))
    print("标准差:", np.std(Auc, ddof=1))

    print("Calculate Auprc's std:")
    print("均值:", np.mean(Auprc))
    print("方差:", np.var(Auprc))
    print("标准差:", np.std(Auprc, ddof=1))

    print("Calculate minpse's std:")
    print("均值:", np.mean(minpse))
    print("方差:", np.var(minpse))
    print("标准差:", np.std(minpse, ddof=1))