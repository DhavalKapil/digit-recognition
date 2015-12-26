import csv
import numpy as np
import h5py
import os
import caffe
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def load_train_set():
  train_images = []
  train_labels = []
  with open('train.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      label = row[0]
      image = np.reshape(row[1:], (1, 28, 28))
      train_images.append(image)
      train_labels.append(label)
  train_images = np.array(train_images, dtype=float)
  train_labels = np.array(train_labels, dtype=float)
  return [train_images, train_labels]

def save_train_set(train_images, train_labels):
  with h5py.File(os.path.join('.', 'train.h5'), 'w') as f:
    f['data'] = train_images
    f['label'] = train_labels
  with h5py.File(os.path.join('.', 'test.h5'), 'w') as f:
    f['data'] = train_images
    f['label'] = train_labels

def train():
  solver = caffe.get_solver('solver.prototxt')
  solver.solve()

def init_net():
  net = caffe.Net('deploy.prototxt', 'image__iter_5000.caffemodel', caffe.TEST)
  return net

def classify(net, input):
  input = np.reshape(input, (1, 1, 28, 28))
  out = net.forward(data=input)
  return np.argmax(out[net.outputs[0]])


caffe.set_mode_cpu()

X, Y = load_train_set()
X, Y = shuffle(X, Y)
X = X / 256

train_images = X[:36000]
train_labels = Y[:36000]

test_images = X[36001:]
test_labels = Y[36001:]

save_train_set(train_images, train_labels)
train()

net = init_net()
i = 0
a1 = []
a2 = []
for input in test_images:
  out_label = classify(net, input)
  a1.append(out_label)
  a2.append(test_labels[i])
  i = i + 1

print accuracy_score(a1, a2)
