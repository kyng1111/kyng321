# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:32:02 2018

@author: KyungMin Park
"""

import os
import tensorflow as tf
from glob import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.python.lib.io import file_io

tf.reset_default_graph()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')


def unpickle(file):
    import pickle
    with file_io.FileIO(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


learning_rate = 0.001;
batch_size = 1000;
EPOCHS = 25;
n_batches = 50;


path = os.path.join(FLAGS.input_dir,"test_batch")
a = unpickle(path)

r = a['data'][:,0:1024].reshape((-1,1))
g = a['data'][:,1024:2048].reshape((-1,1))
b = a['data'][:,2048:3072].reshape((-1,1))
l = np.array(a['labels']).reshape((-1,1))
image1 = np.hstack([r,g,b])
label1 = l



sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('gs://okaygood1/output/output_cifar_model.ckpt-0.meta')
saver.restore(sess,tf.train.latest_checkpoint('gs://okaygood1/output/'))

graph = tf.get_default_graph()

training = graph.get_tensor_by_name("resnet/training:0")
logits = graph.get_tensor_by_name("resnet/fc/logits/MatMul:0")
X = graph.get_tensor_by_name("IteratorGetNext:0")
# difine the target tensor you want evaluate for your prediction
# finally call session to run 


i = random.randint(1,500)

print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: image1[i], training:True}))
print(label1[i])
