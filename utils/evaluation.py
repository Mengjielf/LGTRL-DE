import tensorflow as tf
import keras
from collections import defaultdict
import csv
import json
import numpy as np
import os
import sys
from sklearn.metrics import precision_recall_curve,roc_curve, auc,roc_auc_score
import keras.backend as K
from sklearn import metrics


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


'''for los task'''
class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x and x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None

def get_estimate_custom(prediction, nbins):
    bin_id = np.argmax(prediction)
    assert 0 <= bin_id < nbins
    return CustomBins.means[bin_id]


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100



def print_metrics_regression(y_true, predictions, verbose=1):
    predictions = [get_estimate_custom(x, 10) for x in predictions]
    predictions = np.array(predictions)
    predictions = np.maximum(predictions, 0).flatten()
    y_true = np.array(y_true)

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]

    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("Cohen kappa score = {}".format(kappa))

    return mad,mse,mape,kappa

def print_metrics_regression_rec(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    predictions = np.maximum(predictions, 0).flatten()
    y_true = np.array(y_true)

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("Cohen kappa score = {}".format(kappa))

    return {"mad": mad,
            "mse": mse,
            "mape": mape,
            "kappa": kappa}


def Calculate_Mean_Std_Var_Los(mad,mse,mape,kappa):
    print("Calculate mad's std:")
    print("均值:", np.mean(mad))
    print("方差:", np.var(mad))
    print("标准差:", np.std(mad, ddof=1))

    print("Calculate mse's std:")
    print("均值:", np.mean(mse))
    print("方差:", np.var(mse))
    print("标准差:", np.std(mse, ddof=1))

    print("Calculate mape's std:")
    print("均值:", np.mean(mape))
    print("方差:", np.var(mape))
    print("标准差:", np.std(mape, ddof=1))

    print("Calculate kappa's std:")
    print("均值:", np.mean(kappa))
    print("方差:", np.var(kappa))
    print("标准差:", np.std(kappa, ddof=1))

class LengthOfStayMetrics(keras.callbacks.Callback):
    def __init__(self, x_train, hours_train, x_val, hours_val, batch_size=32,
                 early_stopping=True, verbose=2):
        super(LengthOfStayMetrics, self).__init__()
        self.x_train = x_train
        self.hours_train=hours_train
        self.x_val = x_val
        self.hours_val = hours_val
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, x, hours, history, dataset, logs):
        y_true = []
        predictions = []

        y= hours
        pred = self.model.predict(x, batch_size=self.batch_size)

        y_true += list(y)
        predictions += list(pred)

        predictions = [get_estimate_custom(x, 10) for x in predictions]
        ret = print_metrics_regression_rec(y_true, predictions,self.verbose)

        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        # print("\n==>predicting on train")
        self.calc_metrics(self.x_train, self.hours_train, self.train_history, 'train', logs)
        # print("\n==>predicting on validation")
        self.calc_metrics(self.x_val, self.hours_val, self.val_history, 'val', logs)

        if self.early_stopping:
            max_kappa = np.max([x["kappa"] for x in self.val_history])
            cur_kappa = self.val_history[-1]["kappa"]
            max_train_kappa = np.max([x["kappa"] for x in self.train_history])
            if max_kappa > 0.38 and cur_kappa < 0.35 and max_train_kappa > 0.47:
                self.model.stop_training = True