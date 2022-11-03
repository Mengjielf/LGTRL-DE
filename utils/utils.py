from __future__ import absolute_import
from __future__ import print_function

from . import common_utils
import numpy as np
import os

def save_data(x,y,file_path):
    data=[]
    for i in range(len(x)):
        temp=[]
        temp.append(x[i])
        temp.append(y[i])
        temp=np.array(temp)
        data.append(temp)
    data=np.array(data)
    np.save(file_path,data)

def save_final_data(reader, discretizer, normalizer, type,demographic_data,diagnosis_data,idx_list, save_data_path, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data_x = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    data=data_x
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
        whole_data = (np.array(data), labels)

    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    static_data=[]
    for i in range(len(names)):
        cur_id, cur_ep, _ = names[i].split('_', 2)
        cur_idx = cur_id + '_' + cur_ep
        cur_demo = demographic_data[idx_list.index(cur_idx)]
        cur_diag=diagnosis_data[idx_list.index(cur_idx)]
        cur_static_data=np.hstack((cur_demo,cur_diag))
        static_data.append(cur_static_data)
    static_data=np.array(static_data)

    if type=="train":
        save_data(data, labels, save_data_path+"train.npy")
        np.save(save_data_path+"static_train.npy", static_data)
    if type=="val":
        save_data(data, labels, save_data_path+"val.npy")
        np.save(save_data_path+"static_val.npy", static_data)
    if type=="test":
        save_data(data, labels, save_data_path+"test.npy")
        np.save(save_data_path+"static_test.npy", static_data)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
