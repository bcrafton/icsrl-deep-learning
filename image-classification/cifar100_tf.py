
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'

####################################

import numpy as np
import tensorflow as tf
import keras

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

from keras.layers import Input, Conv2D, BatchNormalization, AveragePooling2D, Dense, Flatten, ReLU, Softmax
from keras.models import Model, Sequential

####################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
y_train = keras.utils.to_categorical(y_train, 100)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
y_test = keras.utils.to_categorical(y_test, 100)

####################################

epochs = 10
batch_size = 50
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 100])

####################################

'''
def block(x, f, p):
    conv1 = tf.layers.conv2d(inputs=x, filters=f, kernel_size=[3, 3], padding='same')
    bn1   = tf.layers.batch_normalization(conv1)
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.layers.conv2d(inputs=relu1, filters=f, kernel_size=[3, 3], padding='same')
    bn2   = tf.layers.batch_normalization(conv2)
    relu2 = tf.nn.relu(bn2)

    pool = tf.layers.max_pooling2d(inputs=relu2, pool_size=[p, p], strides=[p, p], padding='same')
    return pool

block1 = block(x,      128, 2) # 32 -> 16
block2 = block(block1, 256, 2) # 16 -> 8
block3 = block(block2, 512, 2) #  8 -> 4
block4 = block(block3, 512, 4) #  4 -> 1
flat = tf.contrib.layers.flatten(block4)
out = tf.layers.dense(inputs=flat, units=100)
'''

####################################

'''
def batch_norm(x, f, name):
    gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32, name=name+'_gamma')
    beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32, name=name+'_beta')
    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])
    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def block(x, f1, f2, p, name):
    filters = tf.Variable(init_filters(size=[3,3,f1,f2], init='alexnet'), dtype=tf.float32, name=name+'_conv')

    conv = tf.nn.conv2d(x, filters, [1,1,1,1], 'SAME')
    bn   = batch_norm(conv, f2, name+'_bn1')
    relu = tf.nn.relu(bn)

    pool = tf.nn.avg_pool(relu, ksize=[1,p,p,1], strides=[1,p,p,1], padding='SAME')

    return pool

def dense(x, size, name):
    input_size, output_size = size
    w = tf.Variable(init_matrix(size=size, init='alexnet'), dtype=tf.float32, name=name)
    b  = tf.Variable(np.zeros(shape=output_size), dtype=tf.float32, name=name+'_bias')
    fc = tf.matmul(x, w) + b
    return fc

block1 = block(x,        3, 128, 1, 'block1') # 32 -> 32
block2 = block(block1, 128, 128, 2, 'block2') # 32 -> 16
block3 = block(block2, 128, 256, 1, 'block1') # 16 -> 16
block4 = block(block3, 256, 256, 2, 'block1') # 16 -> 8
block5 = block(block4, 256, 512, 1, 'block2') # 8 -> 8
block6 = block(block5, 512, 512, 2, 'block3') # 8 -> 4
block7 = block(block6, 512, 512, 1, 'block2') # 4 -> 4
block8 = block(block7, 512, 512, 4, 'block3') # 4 -> 1
flat   = tf.reshape(block8, [batch_size, 512])
out    = dense(flat, [512, 100], 'fc1')
'''

####################################
'''
predict = tf.argmax(out, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out)
params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).apply_gradients(grads_and_vars)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

####################################

for ii in range(epochs):
    for jj in range(0, 50000, batch_size):
        s = jj
        e = jj + batch_size
        xs = np.reshape(x_train[s:e], (batch_size, 32, 32, 3))
        ys = np.reshape(y_train[s:e], (batch_size, 100))
        sess.run([train], feed_dict={x: xs, y: ys})
        
    total_correct = 0

    for jj in range(0, 10000, batch_size):
        s = jj
        e = jj + batch_size
        xs = np.reshape(x_test[s:e], (batch_size, 32, 32, 3))
        ys = np.reshape(y_test[s:e], (batch_size, 100))
        _sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
        total_correct += _sum_correct

    print ("acc: " + str(total_correct * 1.0 / 10000))
'''
####################################

def block(x, f, p):
    conv = Conv2D(filters=f, kernel_size=[3, 3], strides=[1, 1], padding='same', use_bias=False)(x)
    bn   = BatchNormalization()(conv)
    relu = ReLU()(bn)
    if p == 1:
        return relu
    else:
        pool = AveragePooling2D(pool_size=[p,p], padding='same')(relu)
        return pool

###################

inputs = Input(shape=[32,32,3])

block1 = block(inputs, 128, 1) # 32
block2 = block(block1, 128, 2) # 16

block3 = block(block2, 256, 1) # 16
block4 = block(block3, 256, 2) # 8

block5 = block(block4, 512, 1) # 8
block6 = block(block5, 512, 2) # 4

block7 = block(block6, 512, 1) # 4
block8 = block(block7, 512, 4) # 1

flat   = Flatten()(block8)
fc     = Dense(units=100)(flat)
out    = Softmax()(fc)

model = Model(inputs=inputs, outputs=out)
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1.0), metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

print (model.summary())

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

####################################
'''
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

[out] = sess.run([out], feed_dict={x: x_train[0:50], y: y_train[0:50]})
'''
####################################






