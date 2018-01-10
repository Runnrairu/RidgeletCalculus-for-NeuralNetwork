#事前学習
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import linear_model

def orijinal_adjustment(raw_input,x):#データの加工。組みなおすときにはpandas.Dataframeを使いたい
    long_input=raw_input.shape[0]
    if x==1:
        for i in range(long_input):
            raw_input[i][80]=float(raw_input[i][80])
    for i in range(long_input):
        if raw_input[i][79]=="normal":
            raw_input[i][79]=1.0
        elif raw_input[i][79]=="partial":
            raw_input[i][79]=2.0
        else:
            raw_input[i][79]=0.0
    for i in range(long_input):
        if raw_input[i][78]=="New":
            raw_input[i][78]=1.0
        else :
            raw_input[i][78]=0.0
    for i in range(long_input):
        raw_input[i][77]=2011-float(raw_input[i][77])
    raw_input=np.delete(raw_input,76,1)
    for i in range(long_input):
        raw_input[i][75]=float(raw_input[i][75])
    raw_input=np.delete(raw_input,74,1)
    raw_input=np.delete(raw_input,73,1)
    raw_input=np.delete(raw_input,72,1) #poolQC
    for i in range(long_input):
        raw_input[i][71]=float(raw_input[i][71])
        raw_input[i][70]=float(raw_input[i][70])
        raw_input[i][69]=float(raw_input[i][69])
        raw_input[i][68]=float(raw_input[i][68])
        raw_input[i][67]=float(raw_input[i][67])
        raw_input[i][66]=float(raw_input[i][66])
    raw_input=np.delete(raw_input,65,1)#paved drive
    for i in range(long_input):
        if raw_input[i][64]=="TA":
            raw_input[i][64]=0.0
        elif raw_input[i][64]=="Gd":
            raw_input[i][64]=1.0
        else:
            raw_input[i][64]=-1.0
        if raw_input[i][63]=="TA":
            raw_input[i][63]=0.0
        elif raw_input[i][63]=="Gd":
            raw_input[i][63]=1.0
        else:
            raw_input[i][63]=-1.0
    for i in range(long_input):
        if raw_input[i][62]=="NA":
            raw_input[i][62]=0.0
        else:
            raw_input[i][62]=float(raw_input[i][62])
        if raw_input[i][61]=="NA":
            raw_input[i][61]=0
        else:
            raw_input[i][61]=float(raw_input[i][61])
    raw_input=np.delete(raw_input,60,1)#garagefinish
    for i in range(long_input):
        if raw_input[i][59]=="NA":
            raw_input[i][59]=0.0
        else:
            raw_input[i][59]=2010-float(raw_input[i][59])
    raw_input=np.delete(raw_input,58,1)
    raw_input=np.delete(raw_input,57,1)#fireplaceQu
    for i in range(long_input):
        raw_input[i][56]=float(raw_input[i][56])
    raw_input=np.delete(raw_input,55,1)
    for i in range(long_input):
        raw_input[i][54]=float(raw_input[i][54])
    raw_input=np.delete(raw_input,53,1)
    for i in range(long_input):
        raw_input[i][52]=float(raw_input[i][52])#kitchnAbvGr
        raw_input[i][51]=float(raw_input[i][51])
        raw_input[i][50]=float(raw_input[i][50])
        raw_input[i][49]=float(raw_input[i][49])
        if raw_input[i][48]=="NA":
            raw_input[i][48]=0.0
        else:
            raw_input[i][48]=float(raw_input[i][48])
        if raw_input[i][47]=="NA":
            raw_input[i][47]=0.0
        else:
            raw_input[i][47]=float(raw_input[i][47])
        raw_input[i][46]=float(raw_input[i][46])
        raw_input[i][45]=float(raw_input[i][45])
        raw_input[i][44]=float(raw_input[i][44])
        raw_input[i][43]=float(raw_input[i][43])#1stFlrSF
    raw_input=np.delete(raw_input,42,1)
    raw_input=np.delete(raw_input,41,1)
    raw_input=np.delete(raw_input,40,1)
    raw_input=np.delete(raw_input,39,1)
    for i in range(long_input):
        if raw_input[i][38]=="NA":
            raw_input[i][38]=0.0
        else:
            raw_input[i][38]=float(raw_input[i][38])
        if raw_input[i][37]=="NA":
            raw_input[i][37]=0.0
        else:
            raw_input[i][37]=float(raw_input[i][37])
        if raw_input[i][36]=="NA":
            raw_input[i][36]=0.0
        else:
            raw_input[i][36]=float(raw_input[i][36])
    raw_input=np.delete(raw_input,35,1)#BsmtFinType2
    for i in range(long_input):
        if raw_input[i][34]=="NA":
            raw_input[i][34]=0.0
        else:
            raw_input[i][34]=float(raw_input[i][34])
    raw_input=np.delete(raw_input,33,1)
    raw_input=np.delete(raw_input,32,1)
    raw_input=np.delete(raw_input,31,1)
    raw_input=np.delete(raw_input,30,1)
    raw_input=np.delete(raw_input,29,1)
    raw_input=np.delete(raw_input,28,1)
    raw_input=np.delete(raw_input,27,1)
    for i in range(long_input):
        if raw_input[i][26]=="NA":
            raw_input[i][26]=0.0
        else:
            raw_input[i][26]=float(raw_input[i][26])
    raw_input=np.delete(raw_input,25,1)
    raw_input=np.delete(raw_input,24,1)
    raw_input=np.delete(raw_input,23,1)
    raw_input=np.delete(raw_input,22,1)
    raw_input=np.delete(raw_input,21,1)#RoofStyle
    for i in range(long_input):
        raw_input[i][20]=2010-float(raw_input[i][20])
        raw_input[i][19]=2010-float(raw_input[i][19])
        raw_input[i][18]=float(raw_input[i][18])
        raw_input[i][17]=float(raw_input[i][17])
    raw_input=np.delete(raw_input,16,1)
    raw_input=np.delete(raw_input,15,1)
    raw_input=np.delete(raw_input,14,1)
    raw_input=np.delete(raw_input,13,1)
    raw_input=np.delete(raw_input,12,1)
    raw_input=np.delete(raw_input,11,1)
    raw_input=np.delete(raw_input,10,1)
    raw_input=np.delete(raw_input,9,1)
    raw_input=np.delete(raw_input,8,1)
    raw_input=np.delete(raw_input,7,1)
    raw_input=np.delete(raw_input,6,1)
    raw_input=np.delete(raw_input,5,1)
    for i in range(long_input):
        raw_input[i][4]=float(raw_input[i][4])
        if raw_input[i][3]=="NA":
            raw_input[i][3]=0.0
        else:
            raw_input[i][3]=float(raw_input[i][3])
    raw_input=np.delete(raw_input,2,1)
    for i in range(long_input):
        raw_input[i][1]=float(raw_input[i][1])
    raw_input=np.delete(raw_input,0,1)
    return raw_input

def w_sample():#独特な工夫その２．混合比サンプリング
    w = np.random.rand()
    sum_y_j = 0
    for j in range(data_size):
        sum_y_j += np.absolute(price_train[j][0])
        if w < sum_y_j/y_sum:
            break
    return j
    
    

def oracle_sampling():#独特な工夫その１。オラクルサンプリング。
    a_b_list=[[0 for i in range(CONDITION_SIZE+1)] for j in range(HIDDEN_UNIT_SIZE)]
    
    for j in range(HIDDEN_UNIT_SIZE):
        s=w_sample()
        t=w_sample()
        xi=np.random.beta(100,3)
        gamma=np.random.binomial(1,0.5)
        z=xi*np.power(-1,gamma)
        a_inv=norm(condition_train[s]-condition_train[t])
        norm_s=norm(condition_train[s])
        nor_sum=0 #内積の保存
        for i in range(CONDITION_SIZE):
            a_b_list[j][i] = condition_train[s][i]/(a_inv*norm_s)
            nor_sum +=  a_b_list[j][i]*condition_train[s][i]
            a_b_list[j][CONDITION_SIZE] = nor_sum-z
    return a_b_list


def norm(list):
    norm_sum = 0
    for i in range(list.shape[0]):
        norm_sum += np.power(list[i],2)
    return np.power(norm_sum,0.5)
     
def red_L1():#リッジレット変換後のL1ノルムを近似計算する
    
    


HIDDEN_UNIT_SIZE =10
TRAIN_DATA_SIZE = 1000

raw_input = np.loadtxt(open(r"train.csv"), delimiter=",",skiprows=1,dtype=str)
raw_input=orijinal_adjustment(raw_input,1)
raw_input=raw_input.astype(np.float64)
CONDITION_SIZE = raw_input.shape[1]-1
[condition,price]  = np.hsplit(raw_input, [CONDITION_SIZE])
[condition_train,condition_test]=np.vsplit(condition,[TRAIN_DATA_SIZE])
[price_train,price_test]=np.vsplit(price,[TRAIN_DATA_SIZE])
data_size = price_train.shape[0]
y_sum=0

for i in range(data_size):
    y_sum += np.absolute(float(price_train[i][0]))


para = oracle_sampling() #オラクルサンプリングを行う
[a,b] = np.hsplit(np.array(para), [CONDITION_SIZE])#重みとバイアスに分ける
#実はこの段階で入力層→中間層の学習はほぼ完了している。微妙な近似誤差をtrainingで修正する
#中間層→出力層は線形回帰で近似する


c=clf.coef_ 
d=clf.intercept_

with tf.Graph().as_default():   
    init = tf.global_variables_initializer()
    hidden1_weight = tf.Variable(a, name="hidden1_weight")
    hidden1_bias = tf.Variable(b, name="hidden1_bias")
    output_weight = tf.Variable(c, name="output_weight")
    output_bias = tf.Variable(d, name="output_bias")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess,"../model.ckpt")
