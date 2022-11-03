from __future__ import print_function
import matplotlib
matplotlib.use('Agg')  #一定要把这两行放在最前面
matplotlib.use('PDF')
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam,Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.models import load_model
from keras import regularizers
import numpy as np
import pandas as pd
from sklearn.metrics import *
from model_file import *
from utils.evaluation import *
from utils.DataLoader import *
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve  #画校准曲线


seeds=[1,12,123,1234,12345,1,12,123,1234,12345]

data_path="./data/processed_data/"
x_train,y_train,x_val,y_val,x_test,y_test=data_load(data_path+'train.npy',data_path+'val.npy',data_path+'test.npy')

demographic_train,diagnosis_data_train,demographic_val,diagnosis_data_val,demographic_test,diagnosis_data_test=load_demo(data_path+"static_train.npy",data_path+"static_val.npy",data_path+"static_test.npy")

print(x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_test.shape,y_test.shape)
time_steps=48
input_dims=[76,12]
output_dim=1
batch_size=256


times=5  #设置实验次数
recall=np.zeros(times)
precision=np.zeros(times)
f1_score=np.zeros(times)
Auc=np.zeros(times)
Auprc=np.zeros(times)
minpse=np.zeros(times)
evaluation_avg=np.zeros(6)
for i in range(times):
    np.random.seed(12345)
    emb_size = 128
    trans_emb_size=128
    demo_emb_size=128
    atten_emb_size=128
    d_inner_hid=256
    lstm_units=64
    filepath = "./saved_models/"
    filename = "our_model2_temp_3"+str(i+1)+ ".h5"  #"our_model2_temp_3"+str(i+1)+ ".h5"
    modelpath = filepath + filename
    model = LGTRL_DE(time_steps,4, input_dims, emb_size, trans_emb_size, demo_emb_size,d_inner_hid,lstm_units,output_dim)
    # model.summary()
    optimizer = Adam(lr=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, experimental_run_tf_function = False,metrics=['acc',auroc])

    # checkpoint = ModelCheckpoint(modelpath, monitor='val_auroc', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
    # callbacks_list = [checkpoint]
    # History=model.fit([x_train,demographic_train], y_train, batch_size=batch_size, epochs=40,callbacks=callbacks_list, validation_data=([x_val,demographic_val], y_val),verbose=2)
    #
    # logpath = "./save_history/"
    # logname = "our_model2_temp_5_log"+str(i+1)+ ".pickle"
    # logpath = logpath + logname
    #
    # #保存历史训练log
    # with open(logpath, 'wb') as file_pi:
    #     pickle.dump(History.history, file_pi)
    # loss_plot(History.history)

    # #读取历史训练log
    # with open(logpath, 'rb') as file_pi:
    #     history=pickle.load(file_pi)
    # loss_plot(history)

    # 加载模型
    model.load_weights(modelpath)


    y_pred = model.predict([x_test, demographic_test], batch_size=batch_size, verbose=0)


    recall[i], precision[i], f1_score[i], Auc[i], Auprc[i], minpse[i] = evaluate(y_pred, y_test)
    evaluation_avg[0] += recall[i]
    evaluation_avg[1] += precision[i]
    evaluation_avg[2] += f1_score[i]
    evaluation_avg[3] += Auc[i]
    evaluation_avg[4] += Auprc[i]
    evaluation_avg[5] += minpse[i]

    print('OurModel_2 ', i, ' DONE!')


for j in range(len(evaluation_avg)):
    evaluation_avg[j]=evaluation_avg[j]/times

print("final result:")
print("recall:", evaluation_avg[0], "precision:", evaluation_avg[1], "f1_score:", evaluation_avg[2],
          "Auc:", evaluation_avg[3], "Auprc:", evaluation_avg[4], "minpse:", evaluation_avg[5])

#计算各指标的均值、方差、标准差
Calculate_Mean_Std_Var(recall,precision,f1_score,Auc,Auprc,minpse)