import tensorflow as tf
import numpy as np
import pdb
import os
from datetime import datetime
import slim.net.inception_v1 as inception_v1
from create_tf_record import *
import tensorflow.contrib.slim as slim

from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils
from tensorflow.python import ipu

labels_nums = 5
batch_size = 1
resize_height = 224
resize_width = 224
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

# input_images定义
input_images = tf.placeholder(dtype=tf.float32, shape=[batch_size, resize_height, resize_width, depths], name='input')
# input_labels定义
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, labels_nums], name='label')

# dropout definition
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

train_record_file = 'dataset/record/train224.tfrecords'
val_record_file = 'dataset/record/val224.tfrecords'

train_log_step = 100
base_lr = 0.01
max_steps = 10000
train_param = [base_lr, max_steps]
data_shape = [batch_size, resize_height, resize_width,depths]

val_log_step = 200
snapshot = 2000
snapshot_prefix = 'models/model.ckpt'

# get example_nums of train and val
train_nums=get_example_nums(train_record_file)
val_nums=get_example_nums(val_record_file)
print('train nums:%d,val nums:%d'%(train_nums,val_nums))

# get train data for record
# during training, it needs shuffle=True
# train_images: Tensor("mul:0", shape=(224, 224, 3), dtype=float32)
# train_labels: Tensor("Cast:0", shape=(), dtype=int32)
train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization') # 读取训练数据
print('train_images:%s,train_labels:%s'%(str(train_images), str(train_labels)))

# train_images_batch: Tensor("shuffle_batch:0", shape=(32, 224, 224, 3), dtype=float32)
#  train_labels_batch: Tensor("one_hot:0", shape=(32, 5), dtype=int32)
train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=True)
# during val, shuffle=True is not necessary
val_images, val_labels = read_records(val_record_file, resize_height, resize_width, type='normalization') # 读取验证数据
val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                      batch_size=batch_size, labels_nums=labels_nums,
                                                      one_hot=True, shuffle=False)
def train(input_images, input_labels):
    # Define the model:
    # 导入神经网络模型，获得网络输出
    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        out, end_points = inception_v1.inception_v1(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=keep_prob, is_training=is_training)

    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out) #添加交叉熵损失loss=1.6
    # slim.losses.add_loss(my_loss)
    loss = tf.losses.get_total_loss(add_regularization_losses=False) #添加正则化损失loss=2.2
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))

    # Specify the optimization scheme:
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)


    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9)
    #
    optimizer = tf.train.MomentumOptimizer(learning_rate=base_lr,momentum= 0.9)
    # # train_tensor = optimizer.minimize(loss, global_step)
    # train_op = slim.learning.create_train_op(loss, optimizer,global_step=global_step)


    # 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 更新的过程不包含在正常的训练过程中, 需要我们去手动像下面这样更新
    # 通过`tf.get_collection`获得所有需要更新的`op`
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
    # tf-ipu可以导入slim.learning.create_train_op
    with tf.control_dependencies(update_ops):
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = slim.learning.create_train_op(total_loss=loss,optimizer=optimizer)
        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)
    return train_op, loss, accuracy

with ipu_scope("/device:IPU:0"):
# cost,update = ipu.ipu_compiler.compile(graph,[x,y])
    ipu_run = ipu.ipu_compiler.compile(train, [input_images, input_labels])
    print('ipu_compiler success.')

opts = utils.create_ipu_config()
cfg = utils.auto_select_ipus(opts, 1)
ipu.utils.configure_ipu_system(cfg)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #val_x, val_y = sess.run([val_images_batch, val_labels_batch])
    for i in range(max_steps + 1):
        # 在会话中取出images和labels
        train_x, train_y = sess.run([val_images_batch, val_labels_batch])
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # 这里仅显示每个batch里第一张图片
        #show_image("image", val_x[0, :, :, :])
        print('shape:{},tpye:{},labels:{}'.format(train_x.shape, train_x.dtype, train_y))
        print('shape:{},tpye:{},labels:{}'.format(val_x.shape, val_x.dtype, val_y))
    coord.request_stop()
    coord.join(threads)