

import random

import tensorflow as tf

import os

from glob import glob

import numpy as np

from tensorflow.python.lib.io import file_io





EPOCHS = 10

batch_size = 1000

n_batch = 50

image = np.zeros((1,3072))

label = np.zeros((1,1))

tf.reset_default_graph()



flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', 'input', 'Input Directory.')





def unpickle(file):

    import pickle

    with file_io.FileIO(file,'rb') as fo:

        dict = pickle.load(fo, encoding='latin1')

    return dict





for i in range(1,2):

    path = os.path.join(FLAGS.input_dir,"data_batch_"+str(i))

    a = unpickle(path)

    r = a['data'][:,0:1024].reshape((-1,1))

    g = a['data'][:,1024:2048].reshape((-1,1))

    b = a['data'][:,2048:3072].reshape((-1,1))

    l = np.array(a['labels']).reshape((-1,1))

    image1 = np.hstack([r,g,b]).reshape((-1,3072))

    label1 = l

    image = np.vstack((image1,image))

    label = np.vstack((label1,label))



print(image.shape)

print(label.shape)

