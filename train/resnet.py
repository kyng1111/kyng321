# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:29:44 2018

@author: KyungMin Park
"""

import random
import tensorflow as tf
import os
from glob import glob
import numpy as np


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
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


for i in range(1,6):
    a = unpickle(FLAGS.input_dir+"/data_batch_"+str(i))
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


class resnet:
    def residual_block(self, x, output_channel, downsampling=False, name = 'res_block'):
        if downsampling:
            stride = 2
        else:
            stride = 1
        with tf.variable_scope(name):
            with tf.variable_scope("h1"):
                h1 = tf.layers.conv2d(x,output_channel,[3,3],strides = stride, padding='SAME',use_bias = False,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
                h1 = tf.layers.batch_normalization(h1, training=self.training)
                h1 = tf.nn.relu(h1)
            with tf.variable_scope("h2"):
                h2 = tf.layers.conv2d(h1,output_channel,[3,3],strides = 1, padding='SAME',use_bias = False,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
                h2 = tf.layers.batch_normalization(h2, training=self.training)
            
            if downsampling:
                x = tf.layers.conv2d(x,output_channel,[1,1],strides = stride, padding='SAME',use_bias = False,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
            
            return h2 + x
    
    def build_net(self, x_img, layer_n):
        net = x_img
        
        with tf.variable_scope("conv0"):
            net = tf.layers.conv2d(net,16,[3,3],strides = 1, padding='SAME',use_bias = False,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
            net = tf.layers.batch_normalization(net,training = self.training)
            net = tf.nn.relu(net)
            
        with tf.variable_scope("conv1"):
            for i in range(layer_n):
                net = self.residual_block(net,16,downsampling=False,name='resblock{}'.format(i+1))
            
        with tf.variable_scope("conv2"):
            for i in range(layer_n):
                net = self.residual_block(net,32,downsampling=(i==0),name='resblock{}'.format(i+1))
            
        with tf.variable_scope("conv3"):
            for i in range(layer_n):
                net = self.residual_block(net,64,downsampling=(i==0),name='resblock{}'.format(i+1))

        with tf.variable_scope("fc"):
            net = tf.reduce_mean(net,[1,2])
            logits = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
        return logits
    
    def __init__(self, name = 'resnet',learning_rate =0.001, layer_n=15, SEED=777):
        
        self.x = tf.placeholder(dtype = tf.float32, shape=[None,3072], name = 'x_data')
        self.y = tf.placeholder(dtype = tf.uint8, shape=[None,1], name = 'y_label')
        self.y_onehot = tf.one_hot(self.y,depth=10)
        self.y_onehot1 = tf.reshape(self.y_onehot,([-1,10]))
        self.x_img = tf.reshape(self.x,[-1,32,32,3])
        self.training = tf.placeholder(dtype = tf.bool, name = 'training')
        self.seed = SEED
        
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x_img,self.y_onehot1))
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.shuffle(100000)
        self.dataset = self.dataset.batch(batch_size)
        
        iter = self.dataset.make_initializable_iterator()
        
        self.dataset_init_op = iter.make_initializer(self.dataset,name = 'dataset_init')
        x_image, y_label = iter.get_next()
        
        self.logits = self.build_net(x_image, layer_n = layer_n)
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = y_label))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.dataset_init_op, feed_dict = {self.x : image, self.y : label})
            print("학습시작")
            for i in range(EPOCHS):
                total_cost = 0
                for j in range(n_batch):
                    _, cost_value = sess.run([self.optimizer, self.cost],{self.training:True})
                    total_cost += cost_value
                    print(str(i)+'-'+str(j))
                print("iter:{}, cost:{:.4f}".format(i, total_cost/n_batch))
                    
            saver = tf.train.Saver()
            save_path = saver.save(sess, "./resnet.ckpt")
a = resnet()
