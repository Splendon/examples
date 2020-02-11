import numpy as np
import tensorflow as tf
import time

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils
from tensorflow.python import ipu
#from ipu_utils import get_config

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

t0 = time.time()

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

with ipu_scope("/device:IPU:0"):
    #cost,update = ipu.ipu_compiler.compile(graph,[x,y])
    ipu_run = ipu.ipu_compiler.compile(train, [x, y])

opts = utils.create_ipu_config()
# num_ipus
cfg = utils.auto_select_ipus(opts, num_ipus=4)
ipu.utils.configure_ipu_system(cfg)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training
    for i in range(100000):
        var,L = sess.run(ipu_run, feed_dict={x: rand_x, y: rand_y})
        print('Step #' + str(i+1) + ' A = ' + str(var))
        print('Loss = ' + str(L))

t1 = time.time()
print(t1 - t0)

"""
tensorflow/python/ipu/utils.py
Examples:
.. code-block:: python
  # Create a single device, with one IPU
  opts = create_ipu_config()
  opts = auto_select_ipus(opts, num_ipus=1)
  ipu.utils.configure_ipu_system(opts)
  with tf.Session() as s:
    ...

.. code-block:: python
  # Create two devices, with 2 IPUs per device.
  opts = create_ipu_config()
  opts = auto_select_ipus(opts, num_ipus=[2,2])
  ipu.utils.configure_ipu_system(opts)
  with tf.Session() as s:
    ...
    
.. code-block:: python
  # Create two devices, with 1 IPU in the first device and 2 IPUs
  # in the second device.
  opts = create_ipu_config()
  opts = auto_select_ipus(opts, num_ipus=[1,2])
  ipu.utils.configure_ipu_system(opts)
  with tf.Session() as s:
    ...
"""

"""
globalAMP = None
    if opts["available_memory_proportion"] and len(opts["available_memory_proportion"]) == 1:
        globalAMP = opts["available_memory_proportion"][0]

    ipu_options = get_config(ipu_id=opts["select_ipu"],
                             prng=not opts["no_stochastic_rounding"],
                             shards=opts["shards"],
                             number_of_replicas=opts['replicas'],
                             max_cross_replica_buffer_size=opts["max_cross_replica_buffer_size"],
                             fp_exceptions=opts["fp_exceptions"],
                             xla_recompute=opts["xla_recompute"],
                             seed=opts["seed"],
                             availableMemoryProportion=globalAMP)

    ipu.utils.configure_ipu_system(ipu_options)
    train_sess = tf.Session(graph=train_graph, config=tf.ConfigProto())
"""
