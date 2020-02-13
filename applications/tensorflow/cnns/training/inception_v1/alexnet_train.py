from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

# 定义一个用来显示网络每一层结构的函数print_actications,展示每一个卷积层或池化层输出tensor的尺寸。
# 这个函数接受一个tensor作为输入，并显示其名称（t.op.name）和tensor尺寸（t.get_shape.as_list()）

def print_activation(t):
    print(t.op.name, ' ', t.get_shape().as_list())

# 接下来设计AlexNet的网络结构。并将可训练的参数kernel、biases添加到parameters中
def inference(iamges):
    parameters = []

    # 定义第一个卷积层，并在其后添加LRN层和最大池化层
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(iamges, kernel, strides=[1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                           name='pool1')
    print_activation(pool1)

    # 接下来设计第2个卷积层，大部分步骤和第1个卷积层相同，只有几个参数不同。
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                           name='pool2')
    print_activation(pool2)

    # 定义第3个卷积层
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[384]), trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activation(conv3)

    # 定义第4个卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activation(conv4)

    # 定义第5个卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), trainable=True,
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activation(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                           name='pool5')
    print_activation(pool5)

    return pool5, parameters

# 接下来实现一个评估AlexNet每轮计算时间的函数time_tensorflow_run.第一个输入是Tensorflow的Session，
# 第二个变量是需要测评的运算算子，第三个变量是测试的名称。先定义预热轮数num_steps_burn_in=10,它的作用是
# 给程序预热
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time()
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f + +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches,
                                                                   mn, sd))

# 定义主函数
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3],
                                              dtype=tf.float32, stddev=1e-1))
        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, pool5, "Forward")
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward-backward")

run_benchmark()