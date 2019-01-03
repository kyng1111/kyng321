import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
from glob import glob
from tensorflow.python.lib.io import file_io
import pickle


tf.set_random_seed(777)  # reproducibility

flags=tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir','input','Input Directory.')
flags.DEFINE_string('output_dir','output','Output Directory.')

path = os.path.join(FLAGS.input_dir,"final_input.txt")

print(path)
