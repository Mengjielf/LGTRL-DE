import numpy as np
import os
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedKFold

def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def data_load(train_file,val_file,test_file):
    train=np.load(train_file,allow_pickle=True)
    val=np.load(val_file,allow_pickle=True)
    test=np.load(test_file,allow_pickle=True)
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]
    for i in range(len(train)):
        x_train.append(train[i][0])
        y_train.append(train[i][1])
    for i in range(len(val)):
        x_val.append(val[i][0])
        y_val.append(val[i][1])
    for i in range(len(test)):
        x_test.append(test[i][0])
        y_test.append(test[i][1])
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_val=np.array(x_val)
    y_val=np.array(y_val)
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    return x_train,y_train,x_val,y_val,x_test,y_test


def data_load_single(train_file,val_file,test_file):
    train=np.load(train_file,allow_pickle=True)
    val=np.load(val_file,allow_pickle=True)
    test=np.load(test_file,allow_pickle=True)
    x_train=[]
    x_val=[]
    x_test=[]
    for i in range(len(train)):
        x_train.append(train[i][0])
    for i in range(len(val)):
        x_val.append(val[i][0])
    for i in range(len(test)):
        x_test.append(test[i][0])
    x_train=np.array(x_train)
    x_val=np.array(x_val)
    x_test=np.array(x_test)
    return x_train,x_val,x_test

def data_load_other(train_file,val_file,test_file):
    train=np.load(train_file,allow_pickle=True)
    val=np.load(val_file,allow_pickle=True)
    test=np.load(test_file,allow_pickle=True)
    x_train=[]
    x_val=[]
    x_test=[]
    for i in range(len(train)):
        x_train.append(train[i])
    for i in range(len(val)):
        x_val.append(val[i])
    for i in range(len(test)):
        x_test.append(test[i])
    x_train=np.array(x_train)
    x_val=np.array(x_val)
    x_test=np.array(x_test)
    return x_train,x_val,x_test

def combine(x1,x2,timesteps=5):
    n=len(x1)
    feature1_num=len(x1[0][0])
    feature2_num=len(x2[0][0])
    ans=np.empty([n,5,feature1_num+feature2_num],dtype='uint8')
    for i in range(n):
        for j in range(timesteps):
            seq = np.hstack((x1[i][j], x2[i][j]))
            ans[i][j] = seq
    return ans

def load_demo(train_file,val_file,test_file):
    train=np.load(train_file,allow_pickle=True)
    val=np.load(val_file,allow_pickle=True)
    test=np.load(test_file,allow_pickle=True)
    demographic_train = train[:,0:12]
    diagnosis_data_train = train[:,12:]
    demographic_val = val[:,0:12]
    diagnosis_data_val = val[:,12:]
    demographic_test = test[:,0:12]
    diagnosis_data_test = test[:,12:]

    return demographic_train,diagnosis_data_train,demographic_val,diagnosis_data_val,demographic_test,diagnosis_data_test


def shuffle_data(x,demo,y):
    totalNum=int(len(x))
    index = list(range(totalNum)) #存放下标
    shuffle_x = []
    shuffle_demo = []
    shuffle_y = []
    for i in range(totalNum):
        randomIndex = int(np.random.uniform(0, len(index)))
        shuffle_x.append(x[randomIndex])
        shuffle_demo.append(demo[randomIndex])
        shuffle_y.append(y[randomIndex])
        del index[randomIndex]

    shuffle_x = np.array(shuffle_x)
    shuffle_demo = np.array(shuffle_demo)
    shuffle_y = np.array(shuffle_y)

    return shuffle_x,shuffle_demo,shuffle_y


def convert_5fold_mimic(train_file,val_file,test_file,train_demo_file,val_demo_file,test_demo_file,train_los_file,val_los_file,test_los_file,save_path):
    x_train, y_train, x_val, y_val, x_test, y_test=data_load(train_file,val_file,test_file)
    demographic_train,diagnosis_data_train,demographic_val,diagnosis_data_val,demographic_test,diagnosis_data_test=load_demo(train_demo_file,val_demo_file,test_demo_file)
    los_train,los_val,los_test=data_load_other(train_los_file,val_los_file,test_los_file)

    X=np.concatenate((x_train,x_val,x_test))
    Y=np.concatenate((y_train,y_val,y_test))
    Demo=np.concatenate((demographic_train,demographic_val,demographic_test))
    Los=np.concatenate((los_train,los_val,los_test))

    kf=StratifiedKFold(n_splits=5,shuffle=True, random_state=0)
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(X,Y)):
        save_kfold_mimic(X,Y,Demo,Los,train_idx,test_idx,fold_id,save_path)


def save_kfold_mimic(X,Y,Demo,Los,train_idx,test_idx,kfold,save_path):
    x=X[train_idx]
    test_x=X[test_idx]

    y=Y[train_idx]
    test_y=Y[test_idx]

    demo=Demo[train_idx]
    test_demo=Demo[test_idx]

    los=Los[train_idx]
    test_los=Los[test_idx]

    kf = StratifiedShuffleSplit(n_splits = 1,test_size=0.125)
    for fold_id, (train_index, val_index) in enumerate(kf.split(x,y)):
        val_x=x[val_index]
        val_y=y[val_index]
        val_demo=demo[val_index]
        val_los=los[val_index]

        train_x=x[train_index]
        train_y=y[train_index]
        train_demo=demo[train_index]
        train_los=los[train_index]

    np.save(save_path+str(kfold+1)+"fold_train_x.npy",train_x)
    np.save(save_path+str(kfold+1)+"fold_train_y.npy",train_y)
    np.save(save_path+str(kfold+1)+"fold_train_demo.npy",train_demo)
    np.save(save_path+str(kfold+1)+"fold_train_los.npy",train_los)

    np.save(save_path+str(kfold+1)+"fold_val_x.npy",val_x)
    np.save(save_path+str(kfold+1)+"fold_val_y.npy",val_y)
    np.save(save_path+str(kfold+1)+"fold_val_demo.npy",val_demo)
    np.save(save_path+str(kfold+1)+"fold_val_los.npy",val_los)

    np.save(save_path + str(kfold + 1) + "fold_test_x.npy", test_x)
    np.save(save_path + str(kfold + 1) + "fold_test_y.npy", test_y)
    np.save(save_path + str(kfold + 1) + "fold_test_demo.npy", test_demo)
    np.save(save_path + str(kfold + 1) + "fold_test_los.npy", test_los)

def load_5fold_mimic(x_file,y_file,demo_file,los_file):
    x = np.load(x_file, allow_pickle=True)
    y = np.load(y_file, allow_pickle=True)
    demo = np.load(demo_file, allow_pickle=True)
    los = np.load(los_file, allow_pickle=True)

    return x,y,demo,los

'''
data_path="./data/processed_data/"
convert_5fold_mimic(data_path+'train.npy',data_path+'val.npy',data_path+'test.npy',
                    data_path+"static_train.npy",data_path+"static_val.npy",data_path+"static_test.npy",
                    data_path+"los_train.npy",data_path+"los_val.npy",data_path+"los_test.npy",
                    data_path+"5fold/")
'''
