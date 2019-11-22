

############################################################################

import numpy as np
import tensorflow as tf

############################################################################

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

############################################################################

x_train_1 = np.load('x_train_1.npy')
y_train_1 = np.load('y_train_1.npy')

x_train_2 = np.load('x_train_2.npy')
y_train_2 = np.load('y_train_2.npy')

x_train_3 = np.load('x_train_3.npy')
y_train_3 = np.load('y_train_3.npy')

x_train = np.concatenate((x_train_1, x_train_2, x_train_3), axis=0)
y_train = np.concatenate((y_train_1, y_train_2, y_train_3), axis=0)

x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')

print (type(x_train[0][0][0][0]))
print (type(y_train[0][0][0]))

############################################################################

num_train = np.shape(x_train)[0]
for ii in range(num_train):
    print (ii)
    name = '/home/bcrafton3/Data_SSD/cityscapes/train/%d.tfrecord' % (int(ii))
    with tf.python_io.TFRecordWriter(name) as writer:
        image_raw = x_train[ii].tostring()
        label_raw = y_train[ii].tostring()
        feature={
                'image_raw': _bytes_feature(image_raw),
                'label_raw': _bytes_feature(label_raw)
                }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

############################################################################

num_val = np.shape(x_val)[0]
for ii in range(num_val):
    print (ii)
    name = '/home/bcrafton3/Data_SSD/cityscapes/val/%d.tfrecord' % (int(ii))
    with tf.python_io.TFRecordWriter(name) as writer:
        image_raw = x_val[ii].tostring()
        label_raw = y_val[ii].tostring()
        feature={
                'image_raw': _bytes_feature(image_raw),
                'label_raw': _bytes_feature(label_raw)
                }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

############################################################################


