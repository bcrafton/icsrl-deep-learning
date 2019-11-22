

####################################

import numpy as np
import mxnet as mx
import tensorflow as tf
from gluoncv.data import CitySegmentation

####################################

train_dataset = CitySegmentation(split='train')
train_examples = len(train_dataset)

val_dataset = CitySegmentation(split='val')
val_examples = len(val_dataset)

####################################

name = 1; xs = []; ys = []
for ii in range(train_examples):
    print (ii)
    x, y = train_dataset[ii]
    xs.append(x.asnumpy())
    ys.append(y.asnumpy())
    if ((ii + 1) % 1000 == 0):
        xs = np.stack(xs, axis=0)
        np.save('x_train_%d' % (name), xs)
        ys = np.stack(ys, axis=0)
        np.save('y_train_%d' % (name), ys)
        name = name + 1; xs = []; ys = []

xs = np.stack(xs, axis=0)
np.save('x_train_%d' % (name), xs)
ys = np.stack(ys, axis=0)
np.save('y_train_%d' % (name), ys)

####################################

xs = []; ys = []
for ii in range(val_examples):
    print (ii)
    x, y = val_dataset[ii]
    xs.append(x.asnumpy())
    ys.append(y.asnumpy())

xs = np.stack(xs, axis=0)
np.save('x_val', xs)
ys = np.stack(ys, axis=0)
np.save('y_val', ys)

####################################







