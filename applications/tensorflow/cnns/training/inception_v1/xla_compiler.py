import tensorflow as tf

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.contrib.compiler.xla import xla
from tensorflow.python.ipu import utils
from tensorflow.python import ipu
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


def while_loop(input):
    c = tf.constant(0)
    def body(i):
        return i+1
    cond = lambda i: i < 10
    loop = tf.while_loop(cond, body, [c])

    square = input * input
    # Returned values must be [tensors,] + [operations,]
    # The operations will be used as tf.control_dependencies on the output tensors
    return loop, square, tf.no_op()

with ipu_scope("/device:IPU:0"):
    # Placeholders must be defined outside of xla.compile
    input = tf.placeholder(tf.int32)

    # xla.compile(computation_fn, [inputs])
    out = xla.compile(while_loop, [input])

    # The no_op is not returned and instead added as a control dependency to out.
    assert len(out) == 2

with tf.device('cpu'):
    # Create report gather handle
    report = gen_ipu_ops.ipu_event_trace()

opts = utils.create_ipu_config()
cfg = utils.auto_select_ipus(opts,1)
#ipu.utils.configure_ipu_system(cfg)

with tf.Session(config=tf.ConfigProto(ipu_options=cfg)) as sess:
    sess.run(tf.global_variables_initializer())

    should_be_10, should_be_25 = sess.run(out, feed_dict={input:5})

    assert should_be_10 == 10
    assert should_be_25 == 25

#tensorflow.python.ipu.configure_ipu_system(ipu_options)
#ValueError: Protocol message ConfigProto has no "ipu_options" field.