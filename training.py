
import numpy as np
import tensorflow as tf
import pandas as pd

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



def inference(condition_placeholder):
  with tf.name_scope("hidden1") as scope:
    hidden1_weight = tf.Variable(tf.truncated_normal([CONDITION_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="hidden1_weight")
    hidden1_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="hidden1_bias")
    hidden1_output = tf.nn.relu(tf.matmul(condition_placeholder, hidden1_weight) + hidden1_bias)
  with tf.name_scope("output") as scope:
    output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, 1], stddev=0.1), name="output_weight")
    output_bias = tf.Variable(tf.constant(0.1, shape=[1]), name="output_bias")
    output = tf.matmul(hidden1_output, output_weight) + output_bias
  return tf.nn.l2_normalize(output, 0)

def loss(output, price_placeholder):
  with tf.name_scope("loss") as scope:
    loss = tf.nn.l2_loss(output - tf.nn.l2_normalize(price_placeholder, 0))
    tf.summary.scalar("loss", loss)
  return loss

def training(loss):
  with tf.name_scope("training") as scope:
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
  return train_step


HIDDEN_UNIT_SIZE =10
TRAIN_DATA_SIZE = 1000
raw_input = np.loadtxt(open(r"train.csv"), delimiter=",",skiprows=1,dtype=str)

raw_input=orijinal_adjustment(raw_input,1)
CONDITION_SIZE = raw_input.shape[1]-1
[condition,price]  = np.hsplit(raw_input, [CONDITION_SIZE])
[condition_train,condition_test]=np.vsplit(condition,[TRAIN_DATA_SIZE])
[price_train,price_test]=np.vsplit(price,[TRAIN_DATA_SIZE])


with tf.Graph().as_default():
  condition_placeholder = tf.placeholder("float", [None, CONDITION_SIZE], name="condition_placeholder")
  price_placeholder = tf.placeholder("float", [None, 1], name="price_placeholder")
  loss_label_placeholder = tf.placeholder("string", name="loss_label_placeholder")
  feed_dict_train={
    price_placeholder: price_train,
    condition_placeholder: condition_train,
    loss_label_placeholder: "loss_train",
  }
  feed_dict_test={
    price_placeholder: price_test,
    condition_placeholder: condition_test,
    loss_label_placeholder: "loss_test",
  }

  output = inference(condition_placeholder)
  loss = loss(output, price_placeholder)
  training_op = training(loss)
  summary_op = tf.summary.merge_all()
  init = tf.global_variables_initializer()
  saver=tf.train.Saver()

  with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter('data',graph=sess.graph )
      sess.run(init)
      for step in range(1000):
          sess.run(training_op, feed_dict=feed_dict_train)
          loss_test = sess.run(loss, feed_dict=feed_dict_test)
          if step % 100==0:
              summary_str = sess.run(summary_op, feed_dict_test)
              summary_str += sess.run(summary_op, feed_dict=feed_dict_train)
              summary_writer.add_summary(summary_str, step)
              print(sess.run(loss, feed_dict=feed_dict_train))
      print(sess.run(loss, feed_dict=feed_dict_test))
      saver.save(sess,"../model.ckpt")

