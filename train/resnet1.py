
import random
import tensorflow as tf
import os
from glob import glob
import numpy as np
from tensorflow.python.lib.io import file_io


EPOCHS = 10
batch_size = 1000
n_batches = 50
image = np.zeros((1,3072))
label = np.zeros((1,1))
learning_rate = 0.001
tf.reset_default_graph()

keep_prob=tf.placeholder_with_default(1.0, shape=())
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')


def unpickle(file):
    import pickle
    with file_io.FileIO(file,'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

for i in range(1,6):
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



class ResNet:
    
    # 여기서 x는 net으로 받는다.
    def residual_block(self, x, output_channel, downsampling=False, name='res_block'):
        input_channel = int(x.shape[-1]) #?
    
        if downsampling:
            stride = 2
        else:
            stride = 1
        
        with tf.variable_scope(name):
            with tf.variable_scope('conv1_in_block'):
                # layers ( input , filters 개수, ksize , use_bias, kerner_initializer...)
                h1 = tf.layers.conv2d(x, output_channel, [3,3], strides=stride, padding='SAME', use_bias = False,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))

                h1 = tf.layers.batch_normalization(h1, training=self.training)
                h1 = tf.nn.relu(h1)

            with tf.variable_scope('conv2_in_block'):
                h2 = tf.layers.conv2d(h1, output_channel, [3,3], strides=1, padding='SAME', use_bias=False,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))

                h2 = tf.layers.batch_normalization(h2, training=self.training)

            if downsampling: # use_bias 사용?
                x = tf.layers.conv2d(x, output_channel, [1,1], strides=stride, padding='SAME',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))

            return tf.nn.relu(h2 + x)

    def build_net(self, x_img, layer_n):
        net = x_img
        
        with tf.variable_scope("conv0"):
            net = tf.layers.conv2d(net, 16, [3,3], strides=1, padding='SAME', use_bias=False,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed))
            net = tf.layers.batch_normalization(net, training=self.training)
            net = tf.nn.relu(net)
            
        with tf.variable_scope("conv1"):
            for i in range(layer_n):
                net = self.residual_block(net, 16, name="resblock{}".format(i+1)) ##
                assert net.shape[1:] == [32, 32, 16]
                
        with tf.variable_scope("conv2"):
            for i in range(layer_n):
                net = self.residual_block(net,32,downsampling=(i==0), name="resblock{}".format(i+1))
                assert net.shape[1:] == [16, 16, 32]
       
        with tf.variable_scope("conv3"):
            for i in range(layer_n):
                net = self.residual_block(net, 64, downsampling=(i==0), name="resblock{}".format(i+1))
                assert net.shape[1:] == [8, 8, 64]
                
        with tf.variable_scope("fc"):

            net = tf.reduce_mean(net, [1,2]) 
            assert net.shape[1:] == [64]
            
            logits = tf.layers.dense(net, 10, 
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=self.seed), name="logits")

        return logits
    
    def __init__(self, name='resnet', learning_rate=0.001, layer_n=100, SEED=777):
        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, [None,3072], name='x_data')  # 데이터(사진)가 담길 그릇 32x32x3
            x_img = tf.reshape(self.x,[-1, 32, 32, 3]) 
            self.y = tf.placeholder(tf.uint8, [None,1], name='y_data') # 라벨이 담길 그릇
            y_onehot = tf.one_hot(self.y,depth=10) #. 즉 3은 0,0,0,3,0,0,0..이런식으로
            y_onehot1 = tf.reshape(y_onehot,[-1,10]) # 0~9
            self.training = tf.placeholder(tf.bool, name='training')
            self.seed = SEED            
            
            dataset = tf.data.Dataset.from_tensor_slices((x_img, y_onehot1)) # 데이터와 라벨을 담을 dataset

            dataset = dataset.repeat()
            dataset = dataset.shuffle(100000)
            dataset = dataset.batch(batch_size)

            iter = dataset.make_initializable_iterator() 

            self.dataset_init_op = iter.make_initializer(dataset, name='dataset_init')

            x_image, y_label = iter.get_next()
            
            logits = self.build_net(x_image, layer_n=layer_n)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y_label)) # logits를 크로스엔트로피를 이용하며 softmax 한다.

            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost) # 코스트(로스)값 최소화하기.

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                feed_dict={ self.x: image, self.y: label, keep_prob:0.7 }

                sess.run(self.dataset_init_op, feed_dict=feed_dict) #dataset에 x,y값 전달한다. 

                #train_data_list는 이미지 array고, 이것을 (-1,3072)로 리쉐이프, 이것을 placeholder에 담은 뒤, 다시 x_image로 reshape하고 레이어에 넣는식임.

                print('Training...')
                for i in range(EPOCHS):
                    tot_cost = 0
                    for _ in range(n_batches):
                        _, cost_value = sess.run([self.optimizer, self.cost],{self.training:True})
                        tot_cost += cost_value
                    print("Iter: {}, Loss: {:.4f}".format(i, tot_cost / n_batches)) # 한 배치당 cost

                saver = tf.train.Saver()
                saver.save(sess, "gs://okaygood1/output/cifar_model.ckpt", global_step=0)
                print("finished")



a = ResNet()
