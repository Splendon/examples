import tensorflow as tf
import numpy as np
import pdb
import os

from create_tf_record import *

train_record_file = 'dataset/record/train224.tfrecords'
val_record_file = 'dataset/record/val224.tfrecords'

base_lr = 0.01
batch_size = 32
labels_nums = 5
max_steps = 100
resize_height = 224
resize_width = 224

input_images = tf.placeholder(dtype=tf.float32, shape=[batch_size, resize_height, resize_width, 3], name='input')
input_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, labels_nums], name='label')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

train_nums = get_example_nums(train_record_file)
val_nums = get_example_nums(val_record_file)
print('train nums:%d,val nums:%d' % (train_nums, val_nums))

# get train data for record
# during training, it needs shuffle=True
train_images, train_labels = read_records(train_record_file, 224, 224, type='normalization')
train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=True)
# train_images: Tensor("mul:0", shape=(224, 224, 3), dtype=float32)
# train_labels: Tensor("Cast:0", shape=(), dtype=int32)
print(train_images_batch, train_labels_batch)


# during val, shuffle=True is not necessary
val_images, val_labels = read_records(val_record_file, 224, 224, type='normalization')
val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                      batch_size=batch_size, labels_nums=labels_nums,
                                                      one_hot=True, shuffle=False)

print(val_images_batch, val_labels_batch)
#val_x, val_y = sess.run([val_images_batch, val_labels_batch])

def batch_test(record_file,resize_height, resize_width):
    # 读取record函数
    tf_image,tf_label = read_records(record_file,resize_height,resize_width,type='normalization')
    image_batch, label_batch= get_batch_images(tf_image,tf_label,batch_size=batch_size,labels_nums=labels_nums,one_hot=False,shuffle=False)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:  # 开始一个会话
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(max_steps + 1):
            # 在会话中取出images和labels
            images, labels = sess.run([image_batch, label_batch])
            # 这里仅显示每个batch里第一张图片
            #show_image("image", images[0, :, :, :])
            print('images.shape:{},images.tpye:{}'.format(images.shape,images.dtype))
            print('labels.shape:{},labels.tpye:{},labels:{}'.format(labels.shape, labels.dtype, labels))

        # 停止所有线程
        coord.request_stop()
        coord.join(threads)

batch_test(train_record_file, 224, 224)
