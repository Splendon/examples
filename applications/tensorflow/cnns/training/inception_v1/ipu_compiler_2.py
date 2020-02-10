# Classification
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2

import numpy as np
import tensorflow as tf

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils
from tensorflow.python import ipu

# datalist
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
#x_data = tf.placeholder(shape=[1], dtype=tf.float32)
#y_target = tf.placeholder(shape=[1], dtype=tf.float32)
x = tf.placeholder(shape=[1], dtype=tf.float32)
y = tf.placeholder(shape=[1], dtype=tf.float32)

def train(x, y):
    #x = x_data, y = y_target
    # bias
    #A = tf.Variable(tf.random_normal(mean=10, shape=[1]))
    A = tf.get_variable(initializer = lambda: tf.random_normal(shape=[1], dtype=tf.float32), name = "A")

    # Want to create the operstion sigmoid(x + A)
    # Note, the sigmoid() part is in the loss function
    my_output = tf.add(x, A)

    # expand_dims of expectation
    my_output_expanded = tf.expand_dims(my_output, 0)
    y_target_expanded = tf.expand_dims(y, 0)

    # cross entropy
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)

    # Optimizer
    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(xentropy)
    return A, my_output, xentropy

with ipu_scope("/device:IPU:0"):
    #cost,update = ipu.ipu_compiler.compile(graph,[x,y])
    ipu_run = ipu.ipu_compiler.compile(train, [x, y])

opts = utils.create_ipu_config()
cfg = utils.auto_select_ipus(opts,1)
ipu.utils.configure_ipu_system(cfg)

# Create session
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # training
    for i in range(1400):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]

        bias, output, loss = sess.run(ipu_run, feed_dict={x: rand_x, y: rand_y})
        if (i+1)%10==0:
            print('Step #' + str(i+1) + ' A = ' + str(bias))
            print('Loss = ' + str(loss))
