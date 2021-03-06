import os.path
import tensorflow as tf

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')


x = tf.placeholder("float") # Create a placeholder 'x'
w = tf.Variable(5.0, name="weights")
y = tf.multiply(w, x)

with tf.Session() as sess:
    # Add the variable initializer Op.
    tf.initialize_all_variables().run()

    print(sess.run(y, feed_dict={x: 1.0}))
    print(sess.run(y, feed_dict={x: 2.0}))

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    checkpoint_file = os.path.join(FLAGS.output_dir, 'RNN')
    saver.save(sess, checkpoint_file,global_step=0)
