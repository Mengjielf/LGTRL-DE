import numpy as np

import os
import random

from utils import utils
from utils.readers import InHospitalMortalityReader
from utils.preprocessing import Discretizer, Normalizer


small_part = False
arg_timestep = 1.0
batch_size = 128
data_path = './data/row_data/' # 原始临床时序数据的路径
demo_path = './data/row_data/demographics/' # 原始人口学数据的路径
save_data_path = './data/test/'  # 处理后数据的保存路径

print("===========正在处理临床时序数据===========")
# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'train'),
                                         listfile=os.path.join(data_path, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'train'),
                                       listfile=os.path.join(data_path, 'val_listfile.csv'),
                                       period_length=48.0)
test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'test'),
                                        listfile=os.path.join(data_path, 'test_listfile.csv'),
                                        period_length=48.0)

discretizer = Discretizer(timestep=arg_timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

# %%
discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = 'ihm_normalizer'
normalizer_state = os.path.join(os.path.dirname(data_path), normalizer_state)
normalizer.load_params(normalizer_state)

print("===========正在处理人口学数据===========")
demographic_data = []
diagnosis_data = []
idx_list = []

# demographic
for cur_name in os.listdir(demo_path):
    cur_id, cur_episode = cur_name.split('_', 1)
    cur_episode = cur_episode[:-4]  # 去掉‘.csv'的标识
    cur_file = demo_path + cur_name

    with open(cur_file, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        if header[0] != "Icustay":
            continue
        cur_data = tsfile.readline().strip().split(',')

    if len(cur_data) == 1:
        cur_demo = np.zeros(12)  # 看得出这里是把人口数据离散为12维数据了
        cur_diag = np.zeros(128)  # 这里就是episode里的128维数据(multi-hot vector)
    else:
        if cur_data[3] == '':
            cur_data[3] = 60.0
        if cur_data[4] == '':
            cur_data[4] = 160
        if cur_data[5] == '':
            cur_data[5] = 60

        cur_demo = np.zeros(12)
        cur_demo[int(cur_data[1])] = 1
        cur_demo[5 + int(cur_data[2])] = 1
        cur_demo[9:] = cur_data[3:6]
        cur_diag = np.array(cur_data[8:], dtype=int)

    demographic_data.append(cur_demo)
    diagnosis_data.append(cur_diag)
    idx_list.append(cur_id + '_' + cur_episode)

for each_idx in range(9, 12):  # 对人口数据的最后四维(Age、Height、Weight、Length of stay)进行标准化处理
    cur_val = []
    for i in range(len(demographic_data)):
        cur_val.append(demographic_data[i][each_idx])
    cur_val = np.array(cur_val)
    _mean = np.mean(cur_val)
    _std = np.std(cur_val)
    _std = _std if _std > 1e-7 else 1e-7
    for i in range(len(demographic_data)):
        demographic_data[i][each_idx] = (demographic_data[i][each_idx] - _mean) / _std

print("===========正在保存预处理数据===========")
n_trained_chunks = 0
train_raw = utils.save_final_data(train_reader, discretizer, normalizer, "train",demographic_data, diagnosis_data, idx_list, save_data_path, small_part, return_names=True)
val_raw = utils.save_final_data(val_reader, discretizer, normalizer, "val", demographic_data, diagnosis_data, idx_list, save_data_path, small_part, return_names=True)
test_raw= utils.save_final_data(test_reader, discretizer, normalizer, "test", demographic_data, diagnosis_data, idx_list, save_data_path, small_part, return_names=True)