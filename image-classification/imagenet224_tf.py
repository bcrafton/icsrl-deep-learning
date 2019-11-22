

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--name', type=str, default='imagenet224')
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

exxact = 0
if exxact:
    val_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/val/'
    train_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/train/'
else:
    val_path = '/usr/scratch/datasets/imagenet224/val/'
    train_path = '/usr/scratch/datasets/imagenet224/train/'

val_labels = './imagenet_labels/validation_labels.txt'
train_labels = './imagenet_labels/train_labels.txt'

##############################################

import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from bc_utils.conv_utils import conv_output_length
from bc_utils.conv_utils import conv_input_length

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

##############################################

def write(text):
    print (text)
    f = open(args.name + '.results', "a")
    f.write(text + "\n")
    f.close()

##############################################

# IMAGENET_MEAN = [123.68, 116.78, 103.94]
IMAGENET_MEAN = [0., 0., 0.]

##############################################

def in_top_k(x, y, k):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)

    _, topk = tf.nn.top_k(input=x, k=k)
    topk = tf.transpose(topk)
    correct = tf.equal(y, topk)
    correct = tf.cast(correct, dtype=tf.int32)
    correct = tf.reduce_sum(correct, axis=0)
    return correct

##############################################

# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label

# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `IMAGENET_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def train_preprocess(image, label):
    crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

    means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)

    return centered_image, label
    

# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `IMAGENET_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

    means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)

    return centered_image, label

##############################################

def get_validation_dataset():
    label_counter = 0
    validation_images = []
    validation_labels = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk(val_path):
        for file in files:
            validation_images.append(os.path.join(val_path, file))
    validation_images = sorted(validation_images)

    validation_labels_file = open(val_labels)
    lines = validation_labels_file.readlines()
    for ii in range(len(lines)):
        validation_labels.append(int(lines[ii]))

    remainder = len(validation_labels) % args.batch_size
    validation_images = validation_images[:(-remainder)]
    validation_labels = validation_labels[:(-remainder)]

    return validation_images, validation_labels
    
def get_train_dataset():

    label_counter = 0
    training_images = []
    training_labels = []

    f = open(train_labels, 'r')
    lines = f.readlines()

    labels = {}
    for line in lines:
        line = line.split(' ')
        labels[line[0]] = label_counter
        label_counter += 1

    f.close()

    print ("building train dataset")

    for subdir, dirs, files in os.walk(train_path):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))
                    training_labels.append(labels[folder])

    remainder = len(training_labels) % args.batch_size
    training_images = training_images[:(-remainder)]
    training_labels = training_labels[:(-remainder)]

    return training_images, training_labels

###############################################################

filename = tf.placeholder(tf.string, shape=[None])
label = tf.placeholder(tf.int64, shape=[None])

###############################################################

val_imgs, val_labs = get_validation_dataset()

val_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
val_dataset = val_dataset.shuffle(len(val_imgs))
val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
val_dataset = val_dataset.map(val_preprocess, num_parallel_calls=4)
val_dataset = val_dataset.batch(args.batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(2)

###############################################################

train_imgs, train_labs = get_train_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
train_dataset = train_dataset.shuffle(len(train_imgs))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=4)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(2)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
features = tf.reshape(features, (-1, 224, 224, 3))
labels = tf.one_hot(labels, depth=1000)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

def batch_norm(x, f, name):
    gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32, name=name+'_gamma')
    beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32, name=name+'_beta')
    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])
    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def block(x, f1, f2, p, name):
    filters = tf.Variable(init_filters(size=[3,3,f1,f2], init='alexnet'), dtype=tf.float32, name=name+'_conv')
    conv = tf.nn.conv2d(x, filters, [1,p,p,1], 'SAME')
    bn   = batch_norm(conv, f2, name+'_bn')
    relu = tf.nn.relu(bn)
    return relu

def mobile_block(x, f1, f2, p, name):
    filters1 = tf.Variable(init_filters(size=[3,3,f1,1], init='alexnet'), dtype=tf.float32, name=name+'_conv_dw')
    filters2 = tf.Variable(init_filters(size=[1,1,f1,f2], init='alexnet'), dtype=tf.float32, name=name+'_conv_pw')

    conv1 = tf.nn.depthwise_conv2d(x, filters1, [1,p,p,1], 'SAME')
    bn1   = batch_norm(conv1, f1, name+'_bn_dw')
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.nn.conv2d(relu1, filters2, [1,1,1,1], 'SAME')
    bn2   = batch_norm(conv2, f2, name+'_bn_pw')
    relu2 = tf.nn.relu(bn2)

    return relu2

###############################################################

batch_size = tf.placeholder(tf.int32, shape=())
lr = tf.placeholder(tf.float32, shape=())

bn     = batch_norm(features, 3, 'bn0')                   # 224

block1 = block(bn, 3, 32, 2, 'block1')                    # 224

block2 = mobile_block(block1, 32, 64, 1, 'block2')        # 112
block3 = mobile_block(block2, 64, 128, 2, 'block3')       # 112

block4 = mobile_block(block3, 128, 128, 1, 'block4')      # 56
block5 = mobile_block(block4, 128, 256, 2, 'block5')      # 56

block6 = mobile_block(block5, 256, 256, 1, 'block6')      # 28
block7 = mobile_block(block6, 256, 512, 2, 'block7')      # 28

block8 = mobile_block(block7, 512, 512, 1, 'block8')      # 14
block9 = mobile_block(block8, 512, 512, 1, 'block9')      # 14
block10 = mobile_block(block9, 512, 512, 1, 'block10')    # 14
block11 = mobile_block(block10, 512, 512, 1, 'block11')   # 14
block12 = mobile_block(block11, 512, 512, 1, 'block12')   # 14

block13 = mobile_block(block12, 512, 1024, 2, 'block13')  # 14
block14 = mobile_block(block13, 1024, 1024, 1, 'block14') # 7

pool   = tf.nn.avg_pool(block14, ksize=[1,7,7,1], strides=[1,7,7,1], padding='SAME')
flat   = tf.reshape(pool, [batch_size, 1024])

mat1   = tf.Variable(init_matrix(size=(1024, 1000), init='alexnet'), dtype=tf.float32, name='fc1')
bias1  = tf.Variable(np.zeros(shape=1000), dtype=tf.float32, name='fc1_bias')
fc1    = tf.matmul(flat, mat1) + bias1

###############################################################

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc1, labels=labels)
correct = tf.equal(tf.argmax(fc1, axis=1), tf.argmax(labels, 1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).minimize(loss)

params = tf.trainable_variables()
weights = {}
for p in params:
    weights[p.name] = p

###############################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

'''
[params] = sess.run([params], feed_dict={})
for p in params:
    print (np.shape(p))
assert (False)
'''

# [w] = sess.run([weights], feed_dict={})
# np.save(args.name, w)

###############################################################

for ii in range(args.epochs):

    print('epoch %d/%d' % (ii, args.epochs))

    ##################################################################

    sess.run(train_iterator.initializer, feed_dict={filename: train_imgs, label: train_labs})

    train_total = 0.0
    train_correct = 0.0
    train_acc = 0.0

    for jj in range(0, len(train_imgs), args.batch_size):

        [_total_correct, _] = sess.run([total_correct, train], feed_dict={handle: train_handle, batch_size: args.batch_size, lr: args.lr})

        train_total += args.batch_size
        train_correct += _total_correct
        train_acc = train_correct / train_total

        if (jj % (100 * args.batch_size) == 0):
            p = "train accuracy: %f" % (train_acc)
            write (p)

    ##################################################################

    sess.run(val_iterator.initializer, feed_dict={filename: val_imgs, label: val_labs})

    val_total = 0.0
    val_correct = 0.0
    val_acc = 0.0

    for jj in range(0, len(val_imgs), args.batch_size):

        [_total_correct] = sess.run([total_correct], feed_dict={handle: val_handle, batch_size: args.batch_size, lr: 0.0})

        val_total += args.batch_size
        val_correct += _total_correct
        val_acc = val_correct / val_total

        if (jj % (100 * args.batch_size) == 0):
            p = "val accuracy: %f" % (val_acc)
            write (p)
    
    [w] = sess.run([weights], feed_dict={})
    w['train_acc'] = train_acc
    w['val_acc'] = val_acc
    np.save(args.name, w)
       
    
    
    
    


