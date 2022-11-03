import numpy as np
import os

def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

'''导入诊断数据'''
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

'''导入不含标签的单个数据'''
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

def combine(x1,x2,timesteps=5):  # 合并Dia+Lab
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

def separate_neg_pos(x,demo,y,path):
    '''
    向下采样所有负序列，使其数目与正序列大致相同。返回正样本，划分的负样本下标,数量
    '''
    pos_num = 0
    neg_num = 0
    x_positive=[]
    demo_positive=[]
    y_positive=[]
    for i in range(len(y)):
        if y[i]==1:
            x_positive.append(x[i])
            demo_positive.append(demo[i])
            y_positive.append(y[i])
            pos_num+=1
        else:
            neg_num+=1
    x_positive=np.array(x_positive)
    demo_positive=np.array(demo_positive)
    y_positive=np.array(y_positive)
    print("pos_num:",pos_num,"\n","neg_num:",neg_num) #pos_num: 1987   neg_num: 12694

    # 将负样本按照正样本的比例将负样本分成neg_num/pos_num份
    partition=(int)(neg_num/pos_num)
    rest=neg_num%pos_num
    every_part_num=pos_num+(int)(rest/partition)  #2115
    print(partition)
    neg_pos=[]
    pt=0 #指向原训练集的指针
    for i in range(partition-1):
        cnt=0
        pos=[]
        while cnt<every_part_num:
            if y[pt]==0:
                cnt+=1
                pos.append(pt)
            pt+=1
        neg_pos.append(pos)

    pos=[]
    while pt<len(y):
        if y[pt]==0:
            pos.append(pt)
        pt+=1
    neg_pos.append(pos)

    neg_pos=np.array(neg_pos)
    for i in range(partition):
        data_path=path+"train_position"+str(i+1)+".npy"
        np.save(data_path,neg_pos[i])

    #最后分成了六份，数量分别是2115 2115 2115 2115 2115 2119

    return x_positive,demo_positive,y_positive,neg_pos,partition

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

def bootstrap(data,demo,label, K):
    n = len(data)
    sample_data=[]
    sample_label=[]
    sample_demo=[]
    for i in range(K):
        idx = np.random.randint(0, n)
        print(idx)
        sample_data.append((data[idx]))
        sample_label.append(label[idx])
        sample_demo.append(demo[idx])
    sample_data=np.array(sample_data)
    sample_demo=np.array(sample_demo)
    sample_label=np.array(sample_label)

    return sample_data,sample_demo,sample_label

def load_eicu_data(train_file,test_file):
    train=np.load(train_file,allow_pickle=True)
    test=np.load(test_file,allow_pickle=True)
    train_cat = []
    train_num = []
    train_y = []
    test_cat = []
    test_num = []
    test_y = []
    for i in range(len(train)):
        train_cat.append(train[i][0])
        train_num.append(train[i][1])
        train_y.append(train[i][2])
    for i in range(len(test)):
        test_cat.append(test[i][0])
        test_num.append(test[i][1])
        test_y.append(test[i][2])
    train_cat=np.array(train_cat)
    train_num=np.array(train_num)
    train_y=np.array(train_y)
    test_cat=np.array(test_cat)
    test_num=np.array(test_num)
    test_y=np.array(test_y)
    return train_cat,train_num,train_y,test_cat,test_num,test_y

def get_static_dynamic(cat,num):
    static=cat[:,:,0:3] #'apacheadmissiondx', 'ethnicity', 'gender'
    temp=num[:,:,0:3] #'admissionheight', 'admissionweight', 'age'
    # temp=temp.reshape(-1,temp.shape[1],1)

    static=np.concatenate([static,temp],axis=2)
    static=static[:,0,:]

    dynamic=cat[:,:,3:7]
    dynamic=np.concatenate([dynamic,num[:,:,0:2]],axis=2)
    dynamic=np.concatenate([dynamic,num[:,:,3:13]],axis=2)

    return dynamic,static



