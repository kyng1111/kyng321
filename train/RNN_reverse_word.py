# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:36:15 2019

@author: Kyungmin Park
"""

import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
from glob import glob
from tensorflow.python.lib.io import file_io
import pickle


def unpickle(file):
    with file_io.FileIO(file,'rb') as fo:
        dict = pickle.load(fo)
    return dict

def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell


tf.reset_default_graph()

tf.set_random_seed(777)  # reproducibility

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')


input_path = os.path.join(FLAGS.input_dir, 'real_final01.txt')
a = unpickle(input_path)
dataX2 = np.array(a).reshape(-1,162)

dataX2 = dataX2[::-1]

print(len(dataX2))

tempX2 = []
tempY2 = []

testX2 = []
testY2 = []

for i in range(0,len(dataX2) - 5):
    _x = dataX2[i:i+5]
    _y = dataX2[i+5][:100]
    tempX2.append(_x)
    tempY2.append(_y)
    
del dataX2
print('1')
testX2 = np.array(tempX2)
testY2 = np.array(tempY2)


print("2")

tf.reset_default_graph()

tf.set_random_seed(777)  # reproducibility

print(testX2.shape)
print(testY2.shape)

data_dim = 162
hidden_size = 5
num_classes = 100
learning_rate = 0.01

batch_size = len(testX2)


with tf.variable_scope("rnn1"):
    X = tf.placeholder(tf.float32, [None,5,data_dim],name = 'x_data')
    Y = tf.placeholder(tf.float32, [None,num_classes],name = 'y_data')

    dataset = tf.data.Dataset.from_tensor_slices((X,Y))
    dataset = dataset.repeat()
    dataset = dataset.batch(10000)

    iter = dataset.make_initializable_iterator()
    dataset_init_op = iter.make_initializer(dataset,name='dataset_init')

    X_, Y_ = iter.get_next()


    multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(5)], state_is_tuple=True)

    # outputs: unfolding size x hidden size, state = hidden size
    outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_, dtype=tf.float32)

    # FC layer
    outputs = tf.contrib.layers.fully_connected(outputs[:,-1], num_classes, activation_fn=tf.nn.tanh)

    outputs1 = tf.convert_to_tensor(outputs,name="y_pred")



    ## sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)

    sequence_loss = tf.reduce_sum(tf.square(outputs1 - Y_))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(sequence_loss)
with tf.Session() as sess:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(dataset_init_op, feed_dict = {X:testX2, Y:testY2})
    print("starts")
    for i in range(1000):
        tot_cost = 0
        for j in range(156):
            _, l, results = sess.run([train_op, sequence_loss, outputs])
            tot_cost += l
        if i==999:
            print("Iter: {}, Loss: {:.4f}".format(i, tot_cost))
        
    saver = tf.train.Saver()    
    checkpoint_file = os.path.join(FLAGS.output_dir, 'RNN_reverse_word')
    saver.save(sess, checkpoint_file,global_step=0)
    
'''   
# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: testX[3]})
for j, result in enumerate(results):
    if j is 0:  # print all for the first result to make a sentence
        print(results)
    else:
        print(results)
'''
