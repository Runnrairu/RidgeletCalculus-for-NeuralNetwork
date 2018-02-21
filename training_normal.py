import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def inference(condition_placeholder,keep_prob):
  with tf.name_scope("hidden1") as scope:
    hidden1_output = tf.nn.relu(tf.matmul(condition_placeholder, hidden1_weight) + hidden1_bias)
    hid1_output = tf.nn.dropout(hidden1_output,keep_prob)
  with tf.name_scope("output") as scope:  
    output = Z*(tf.matmul(hid1_output, output_weight) + output_bias)
    drop_output = tf.nn.dropout(output,keep_prob)
  return tf.nn.l2_normalize(output, 0)

def loss(output, label_placeholder):
  with tf.name_scope("loss") as scope:
    loss = tf.nn.l2_loss(output - tf.nn.l2_normalize(label_placeholder, 0))
    tf.summary.scalar("loss", loss)
  return loss

def training(loss):
  with tf.name_scope("training") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss,var_list=[hidden1_weight,hidden1_bias,output_weight,output_bias])
  return train_step

def training_pre(loss):
  with tf.name_scope("training") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss,var_list = [Z])
  return train_step



HIDDEN_UNIT_SIZE =1000
TRAIN_DATA_SIZE = 1000
raw_input = np.loadtxt(open(r"train.csv"), delimiter=",",skiprows=1,dtype = 'float')
CONDITION_SIZE = raw_input.shape[1]-1
[condition,label]  = np.hsplit(raw_input, [CONDITION_SIZE])
[condition_train,condition_test]=np.vsplit(condition,[TRAIN_DATA_SIZE])
[label_train,label_test]=np.vsplit(label,[TRAIN_DATA_SIZE])
losstrain = []
losstest = []

with tf.Graph().as_default():
  condition_placeholder = tf.placeholder("float", [None, CONDITION_SIZE], name="condition_placeholder")
  label_placeholder = tf.placeholder("float", [None, 1], name="label_placeholder")
  loss_label_placeholder = tf.placeholder("string", name="loss_label_placeholder")
  keep_prob = tf.placeholder("float")
  feed_dict_train={
    label_placeholder: label_train,
    condition_placeholder: condition_train,
    loss_label_placeholder: "loss_train",
    keep_prob : 0.5
  }
  feed_dict_test={
    label_placeholder: label_test,
    condition_placeholder: condition_test,
    loss_label_placeholder: "loss_test",
    keep_prob : 1.0
  }
  hidden1_weight = tf.Variable(tf.truncated_normal([CONDITION_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="hidden1_weight")
  hidden1_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="hidden1_bias")
  output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, 1], stddev=0.1), name="output_weight")
  output_bias = tf.Variable(tf.constant(0.1, shape=[1]), name="output_bias")  
  Z = tf.Variable(1.0, name="Z") 
  output = inference(condition_placeholder,keep_prob)
  loss = loss(output, label_placeholder)
  pretraining = training_pre(loss)
  training_op = training(loss)
  summary_op = tf.summary.merge_all()
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter('data',graph=sess.graph )
      sess.run(init)
      #for step in range(10):
      #    sess.run(pretraining, feed_dict=feed_dict_train)
      for step in range(10000):
          sess.run(training_op, feed_dict=feed_dict_train)
          loss_test = sess.run(loss, feed_dict=feed_dict_test)
          loss_train = sess.run(loss, feed_dict=feed_dict_train)
          losstrain.append(loss_train)
          losstest.append(loss_test)
          if step % 100==0:
              summary_str = sess.run(summary_op, feed_dict_test)
              summary_str += sess.run(summary_op, feed_dict=feed_dict_train)
              summary_writer.add_summary(summary_str, step)
              print(loss_train)       
      print(sess.run(loss, feed_dict=feed_dict_test))
plt.plot(losstrain,label="train")
plt.plot(losstest,label="test") 
plt.legend()
plt.title("loss")
plt.xlabel("step")
plt.ylabel("L^2-loss")
plt.show()
      
