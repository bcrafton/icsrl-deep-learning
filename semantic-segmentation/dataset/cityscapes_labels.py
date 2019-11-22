
# following this tutorial to check out city scapes.
# https://gluon-cv.mxnet.io/build/examples_datasets/cityscapes.html

from gluoncv.data import CitySegmentation
train_dataset = CitySegmentation(split='train')
val_dataset = CitySegmentation(split='val')
print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))

#########################

import numpy as np
img, mask = val_dataset[0]

print (np.max(mask))

#########################

print (np.shape(img))
print (np.shape(mask))

labels = []
[x, y] = np.shape(mask)
for ii in range(x):
    for jj in range(y):
        label = mask[ii][jj]
        if label not in labels:
            labels.append(label)

print (labels)

#########################
