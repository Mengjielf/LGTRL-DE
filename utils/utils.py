from __future__ import absolute_import
from __future__ import print_function

from . import common_utils
import numpy as np
import os

def save_data(x,y,file_path):#保存训练、验证、测试数据
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
    names = ret["name"]  #类似‘17774_episode1_timeseries.csv’
    data_x = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    data=data_x
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    #到这里就是完整的处理后的数据(shape=(48,76))及标签了   对于shape,所有的数据都是48*76,48是因为入院死亡率统计的是入院后48小时的情况，所以之前应该是做了一些处理，所以留下来的数据都是48h的
    whole_data = (np.array(data), labels)

    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    static_data=[]
    for i in range(len(names)):
        cur_id, cur_ep, _ = names[i].split('_', 2)  # batch_name[i]类似于“96468_episode1_timeseries.csv”
        cur_idx = cur_id + '_' + cur_ep
        # print(cur_idx)  #类似’subjectid'+'_'+'episode'+num,如： 96468_episode1
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
