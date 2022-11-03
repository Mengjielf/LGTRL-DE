from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import matplotlib
matplotlib.use('Agg')
matplotlib.use('PDF')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from model_file import *
from utils.evaluation import *
from utils.DataLoader import *
import utils.Config as mimicConfig

time_steps=48
input_dims=[76,12]
output_dim=1
batch_size=256

config=mimicConfig()


times=config.k_fold
recall=np.zeros(times)
precision=np.zeros(times)
f1_score=np.zeros(times)
Auc=np.zeros(times)
Auprc=np.zeros(times)
minpse=np.zeros(times)
evaluation_avg=np.zeros(6)
for j in range(5):
    for i in range(1):
        np.random.seed(config.seed)

        x_train, y_train, demographic_train, los_train = load_5fold_mimic(
            config.data_path + str(i + 1) + "fold_train_x.npy",
            config.data_path + str(i + 1) + "fold_train_y.npy",
            config.data_path + str(i + 1) + "fold_train_demo.npy",
            config.data_path + str(i + 1) + "fold_train_los.npy")
        x_val, y_val, demographic_val, los_val = load_5fold_mimic(config.data_path + str(i + 1) + "fold_val_x.npy",
                                                                  config.data_path + str(i + 1) + "fold_val_y.npy",
                                                                  config.data_path + str(i + 1) + "fold_val_demo.npy",
                                                                  config.data_path + str(i + 1) + "fold_val_los.npy")
        x_test, y_test, demographic_test, los_test = load_5fold_mimic(config.data_path + str(i + 1) + "fold_test_x.npy",
                                                                      config.data_path + str(i + 1) + "fold_test_y.npy",
                                                                      config.data_path + str(
                                                                          i + 1) + "fold_test_demo.npy",
                                                                      config.data_path + str(
                                                                          i + 1) + "fold_test_los.npy")
        emb_size = 128
        trans_emb_size = emb_size
        demo_emb_size = emb_size
        atten_emb_size = emb_size
        d_inner_hid = emb_size * 2
        lstm_units = int(emb_size / 2)

        filepath = config.model_dir
        filename = "lgtrl_de_" + str(i + 1) + str(j + 1) + ".h5"
        modelpath = filepath + filename

        model = LGTRL_DE(time_steps, 4, input_dims, emb_size, trans_emb_size, demo_emb_size, d_inner_hid, lstm_units,output_dim)
        optimizer = Adam(lr=0.0005)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, experimental_run_tf_function=False,
                      metrics=['acc', auroc])

        model.load_weights(modelpath)

        y_pred = model.predict([x_test, demographic_test], batch_size=batch_size, verbose=0)

        recall[i], precision[i], f1_score[i], Auc[i], Auprc[i], minpse[i] = evaluate(y_pred, y_test)
        evaluation_avg[0] += recall[i]
        evaluation_avg[1] += precision[i]
        evaluation_avg[2] += f1_score[i]
        evaluation_avg[3] += Auc[i]
        evaluation_avg[4] += Auprc[i]
        evaluation_avg[5] += minpse[i]


for j in range(len(evaluation_avg)):
    evaluation_avg[j]=evaluation_avg[j]/times

print("final result:")
print("recall:", evaluation_avg[0], "precision:", evaluation_avg[1], "f1_score:", evaluation_avg[2],
          "Auc:", evaluation_avg[3], "Auprc:", evaluation_avg[4], "minpse:", evaluation_avg[5])

Calculate_Mean_Std_Var(recall,precision,f1_score,Auc,Auprc,minpse)