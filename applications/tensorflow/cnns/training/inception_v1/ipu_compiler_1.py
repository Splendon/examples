import numpy as np
import tensorflow as tf

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils
from tensorflow.python import ipu

# Regression example:
# We will create sample data as follows:
# x-data: 100 random samples from a normal ~ N(1, 0.1)
# target: 100 values of the value 10.
# We will fit the model:
# x-data * A = target
# Theoretically, A = 10.

# make data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

rand_index = np.random.choice(100)
rand_x = [x_vals[rand_index]]
rand_y = [y_vals[rand_index]]
x = tf.placeholder(shape=[1], dtype=tf.float32)
y = tf.placeholder(shape=[1], dtype=tf.float32)

def train(x, y):
    # Create variable (one model parameter = A)
#    A = tf.Variable(tf.random_normal(shape=[1]))
    A = tf.get_variable(initializer=lambda: tf.random_normal(shape=[1], dtype=tf.float32), name="A")

    my_output = tf.multiply(x, A)
    #my_output = tf.multiply(x, y)

    # L2 loss
    loss = tf.square(my_output - y)

    # Optimizer
    my_opt = tf.train.GradientDescentOptimizer(0.02)
    train_step = my_opt.minimize(loss)
    return A,loss

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training
    for i in range(1000):
        with ipu_scope("/device:IPU:0"):
            # cost,update = ipu.ipu_compiler.compile(graph,[x,y])
            ipu_run = ipu.ipu_compiler.compile(train, [x, y])

        opts = utils.create_ipu_config()
        cfg = utils.auto_select_ipus(opts, 1)
        ipu.utils.configure_ipu_system(cfg)

        var,L = sess.run(ipu_run, feed_dict={x: rand_x, y: rand_y})
        print('Step #' + str(i+1) + ' A = ' + str(var))
        print('Loss = ' + str(L))