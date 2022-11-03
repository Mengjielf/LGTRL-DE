from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  #（代表仅使用第0，1号GPU）
import matplotlib
matplotlib.use('Agg')  #一定要把这两行放在最前面
matplotlib.use('PDF')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from model_file import *
from utils.evaluation import *
from utils.DataLoader import *



data_path="./data/processed_data/"  # 保存的处理后的数据路径
filepath = "./saved_models/"   # 保存的模型存放的路径

x_train,y_train,x_val,y_val,x_test,y_test=data_load(data_path+'train.npy',data_path+'val.npy',data_path+'test.npy')
demographic_train,diagnosis_data_train,demographic_val,diagnosis_data_val,demographic_test,diagnosis_data_test=load_demo(data_path+"static_train.npy",data_path+"static_val.npy",data_path+"static_test.npy")
print("=================模型的输入形状大小=============")
print("临床时序数据的形状： ", x_train.shape, "人口学数据的形状： ", demographic_train.shape, "患者标签的形状： ", y_train.shape)

time_steps=48
input_dims=[76,12]
output_dim=1
batch_size=256


nums=5  # 保存的模型个数
recall=np.zeros(nums)
precision=np.zeros(nums)
f1_score=np.zeros(nums)
Auc=np.zeros(nums)
Auprc=np.zeros(nums)
minpse=np.zeros(nums)
evaluation_avg=np.zeros(6)
for i in range(nums):
    print("=========测试保存的第"+str(i+1)+"个模型,测试结果如下==============")
    emb_size = 128
    trans_emb_size=128
    demo_emb_size=128
    atten_emb_size=128
    d_inner_hid=256
    lstm_units=64
    filename = "our_model_"+str(i+1)+ ".h5"
    modelpath = filepath + filename
    model = LGTRL_DE(time_steps, 4, input_dims, emb_size, trans_emb_size, demo_emb_size,d_inner_hid,lstm_units,output_dim)

    # 编译模型
    optimizer = Adam(lr=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, experimental_run_tf_function = False,metrics=['acc',auroc])

    # 加载模型
    model.load_weights(modelpath)

    # 获取预测结果
    y_pred = model.predict_on_batch([x_test, demographic_test], batch_size=batch_size, verbose=0)

    # 评估预测结果
    recall[i], precision[i], f1_score[i], Auc[i], Auprc[i], minpse[i] = evaluate(y_pred, y_test)
    evaluation_avg[0] += recall[i]
    evaluation_avg[1] += precision[i]
    evaluation_avg[2] += f1_score[i]
    evaluation_avg[3] += Auc[i]
    evaluation_avg[4] += Auprc[i]
    evaluation_avg[5] += minpse[i]

    print('Model ', i, ' DONE!')


for j in range(len(evaluation_avg)):
    evaluation_avg[j]=evaluation_avg[j]/nums

print("==============final result==============")
print("recall:", evaluation_avg[0], "precision:", evaluation_avg[1], "f1_score:", evaluation_avg[2],
          "Auc:", evaluation_avg[3], "Auprc:", evaluation_avg[4], "minpse:", evaluation_avg[5])

# 计算各指标的均值、方差、标准差
Calculate_Mean_Std_Var(recall,precision,f1_score,Auc,Auprc,minpse)