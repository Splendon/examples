import tensorflow as tf
import numpy as np

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.contrib.compiler.xla import xla
from tensorflow.python.ipu import utils
from tensorflow.python import ipu
from tensorflow.python.ipu.autoshard import automatic_sharding

def graph(x, y):
    with ipu.ops.ipu_shard(0):
        W = tf.get_variable(initializer=lambda:tf.zreos([200, 10]), dtype=tf.float16, name='W')
        b = tf.get_variable(initializer=lambda:tf.zreos([10]), dtype=tf.float16, name='b')
        pred = tf.nn.softmax(tf.matmul(x, W, name='matmul') +b, name='pred')

    # Place the loss on another
    with ipu.ops.ipu_shard(1):
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1), name='cost')

    # sharedOptimizer will explicitly collocate fwd, grad & wu ops.
    optimizer = ipu.sharded_optimizer(tf.train.GradientDescentOptimizer(0.01))

    return cost, optimizer.minimize(cost)

with ipu_scope('/device:IPU:0'):
    x = tf.placeholder(tf.float32, [None, 200], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')

    cost = xla.compile(graph, [x, y])

utils.move_variable_initialization_to_cpu()
init = tf.global_variables_initializer()

opts = ipu.utils.create_ipu_config()
# Select 2 IPUs and select sharded option
ipu.utils.auto_select_ipus(opts, 2, sharded=True)

config = tf.ConfigProto(ipu_options=opts)

with tf.Session(config=config) as sess:
    train_x = np.random.random([50, 200])
    train_y = np.random.random([50, 10])

    sess.run(init)
    c, _ = sess.run([cost], feed_dict={x:train_x, y:train_y})
    print(c)


"""
No work yet.
AttributeError: module 'tensorflow.python.ipu.ops' has no attribute 'ipu_shard'
"""