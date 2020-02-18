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
input_images = tf.placeholder(dtype=tf.float16, shape=[batch_size, resize_height, resize_width, depths], name='input')
# input_labels定义
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, labels_nums], name='label')

# dropout definition
keep_prob = tf.placeholder(tf.float16,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

train_record_file = 'dataset/record/train224.tfrecords'
val_record_file = 'dataset/record/val224.tfrecords'

train_log_step = 20
base_lr = 0.01
max_steps = 10000
train_param = [base_lr, max_steps]

val_log_step = 200
snapshot = 2000
snapshot_prefix = 'models/model.ckpt'

# get example_nums of train and val
train_nums=get_example_nums(train_record_file)
val_nums=get_example_nums(val_record_file)
print('train nums:%d,val nums:%d'%(train_nums,val_nums))

# get train data for record
# during training, it needs shuffle=True

def train(input_images, input_labels):
    # Define the model:
    # 导入神经网络模型，获得网络输出
#    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
    out, end_points = inception_v1.inception_v1(inputs=input_images,
                 num_classes=labels_nums,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1',
                 global_pool=False)

    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)#添加交叉熵损失loss=1.6
    # slim.losses.add_loss(my_loss)
    loss = tf.losses.get_total_loss(add_regularization_losses=False)#添加正则化损失loss=2.2

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))

    #optimizer = tf.train.MomentumOptimizer(learning_rate=base_lr,momentum= 0.9)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)

    # # train_tensor = optimizer.minimize(loss, global_step)
    train_op = slim.learning.create_train_op(loss, optimizer)

    return loss, accuracy, train_op

with ipu_scope("/device:IPU:0"):
# cost,update = ipu.ipu_compiler.compile(graph,[x,y])
    ipu_run = ipu.ipu_compiler.compile(train, [input_images, input_labels])
    print('ipu_compiler success.')

opts = utils.create_ipu_config()
cfg = utils.auto_select_ipus(opts, 1)
ipu.utils.configure_ipu_system(cfg)

def net_evaluation(sess,val_images_input,val_labels_input,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_input, val_labels_input])
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss,val_acc, _ = sess.run(ipu_run, feed_dict={input_images: val_x,input_labels: val_y}) # keep_prob:1.0, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    batch_loss = np.array(val_losses, dtype=np.float32).mean()
    batch_acc = np.array(val_accs, dtype=np.float32).mean()
    return batch_loss, batch_acc

def step_train(train_data, val_data):
    # 训练过程参数保存
    saver = tf.train.Saver()
    max_acc = 0.0

    train_images, train_labels = read_records(train_data, resize_height, resize_width,
                                              type='normalization')  # 读取训练数据
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=True)
    # during val, shuffle=True is not necessary
    val_images, val_labels = read_records(val_data, resize_height, resize_width,
                                          type='normalization')  # 读取验证数据
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)
    # 启动tf.Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # tf协调器和入队线程启动器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps + 1):
            # input dataflow
            train_x, train_y = sess.run([train_images_batch, train_labels_batch])
            train_loss, train_acc, _ = sess.run(ipu_run, feed_dict={input_images: train_x, input_labels: train_y})
#                                                                  keep_prob: 0.8, is_training: True})
#            print(train_loss, train_acc)

            # train for one-batch
            if i % train_log_step == 0:
                print("%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (datetime.now(), i, train_loss, train_acc))

            # val
            if i % val_log_step == 0:
                mean_loss, mean_acc = net_evaluation(sess, val_images_batch, val_labels_batch, val_nums)
                print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))

            # model snapshot
            if (i % snapshot == 0 and i > 0) or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix, i))
                saver.save(sess, snapshot_prefix, global_step=i)
            # save model with highest accuracy in val
            if mean_acc > max_acc and mean_acc > 0.7:
                max_acc = mean_acc
                path = os.path.dirname(snapshot_prefix)
                best_models = os.path.join(path, 'best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
                print('------save:{}'.format(best_models))
                saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    # 循环迭代过程
    step_train(train_record_file, val_record_file)