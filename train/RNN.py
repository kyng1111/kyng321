import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
from glob import glob
from tensorflow.python.lib.io import file_io
import pickle

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def unpickle(file):
    with file_io.FileIO(file,'rb') as fo:
        dict = pickle.load(fo)
    return dict

tf.reset_default_graph()

tf.set_random_seed(777)  # reproducibility

del_all_flags(tf.flags.FLAGS)

flags=tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir','input','Input Directory.')
flags.DEFINE_string('output_dir','output','Output Directory.')

data_dim = 200
hidden_size = 10
num_classes = 200
learning_rate = 0.01
path = os.path.join(FLAGS.input_dir,"final_input.txt")


a = unpickle(path)
dataX = np.array(a).reshape(-1,200)

print(path)




tempX = []
tempY = []

for i in range(0,len(dataX) - 10):
    _x = dataX[i:i+10]
    _y = dataX[i+10]
    tempX.append(_x)
    tempY.append(_y)

testX = np.array(tempX)
testY = np.array(tempY)
print(testX.shape)
print(testY.shape)

batch_size = len(testX)

X = tf.placeholder(tf.float32, [None,10,200],name = 'x_data')
Y = tf.placeholder(tf.float32, [None,200],name = 'y_data')

# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
    
# FC layer
outputs = tf.contrib.layers.fully_connected(outputs[:,-1], num_classes, activation_fn=None)

outputs1 =tf.math.scalar_mul(1, outputs)
outputs2 = tf.convert_to_tensor(outputs1,name="y_pred")



## sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)

sequence_loss = tf.reduce_sum(tf.square(outputs2 - Y)) 

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(sequence_loss)

with tf.Session() as sess:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("학습시작")
    for i in range(500):
        _, l, results = sess.run(
            [train_op, sequence_loss, outputs], feed_dict={X: testX, Y: testY})
        print(i,l)
    saver = tf.train.Saver()
    checkpoint_file = os.path.join(FLAGS.output_dir,'checkpoint')
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
