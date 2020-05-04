#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import tensorflow as tf

from tensorflow.keras.layers import Dense as TFDense, Activation as TFActivation, Flatten as TFFlatten
from keras.layers import Dense as KerasDense, Activation as KerasActivation, Flatten as KerasFlatten
from mnist_cifar_models import get_model_meta, NLayerModel

from setup_mnist import MNIST
from utils import show
from PIL import Image
import argparse

# Models used to load Pytorch model
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# MLP model, each layer has the same number of neuron
# parameter in_dim: input image dimension, 784 for MNIST and 1024 for CIFAR
# parameter layer: number of layers
# parameter neuron: number of neurons per layer
def model_mlp_uniform(in_dim, layer, neurons, out_dim = 10):
    assert layer >= 2
    neurons = [neurons] * (layer - 1)
    return model_mlp_any(in_dim, neurons, out_dim)

# MLP model, each layer has the different number of neurons
# parameter in_dim: input image dimension, 784 for MNIST and 1024 for CIFAR
# parameter neurons: a list of neurons for each layer
def model_mlp_any(in_dim, neurons, out_dim = 10, flatten = True):
    assert len(neurons) >= 1
    # input layer
    if flatten:
        units = [Flatten(), nn.Linear(in_dim, neurons[0])]
    else:
        units = [nn.Sequential(), nn.Linear(in_dim, neurons[0])]
    prev = neurons[0]
    # intermediate layers
    for n in neurons[1:]:
        units.append(nn.ReLU())
        units.append(nn.Linear(prev, n))
        prev = n
    # output layer
    units.append(nn.ReLU())
    units.append(nn.Linear(neurons[-1], out_dim))
    #print(units)
    return nn.Sequential(*units)

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',help='input file', required=True)
parser.add_argument('-o','--output',help='output file', required=True)
parser.add_argument('-f','--flatten',action='store_true',help='add flatten to input')
parser.add_argument('--image_size',default=28,help='Image size, only use when flatten is true')
parser.add_argument('--image_channel',default=1,help='Image channel, only use when flatten is true')
parser.add_argument('neurons', metavar='N', type=int, nargs='+',
                    help='number of neurons for each layer, include input and output layer')
args = parser.parse_args()

neurons = args.neurons
in_dim = neurons[0]
out_dim = neurons[-1]
torch_model = model_mlp_any(in_dim, neurons[1:-1], out_dim, args.flatten)
torch_model.load_state_dict(torch.load(args.input)['state_dict'])

data = MNIST()
n_data = 500
pred_data = data.test_data[:n_data]
pred_label = data.test_labels[:n_data]

if args.flatten:
    keras_model = NLayerModel(neurons[1:-1], image_size = args.image_size, image_channel = args.image_channel, flatten = args.flatten, out_dim = out_dim)
else:
    keras_model = NLayerModel(neurons[1:-1], image_size = in_dim, flatten = args.flatten, out_dim = out_dim)

torch_weights = []
torch_bias = []
for l in torch_model:
    if isinstance(l, torch.nn.Linear):
        torch_weights.append(l.weight)
        torch_bias.append(l.bias)
for i, U in enumerate(keras_model.U):
    U.set_weights([torch_weights[i].t().detach().numpy(), torch_bias[i].detach().numpy()])
keras_model.W.set_weights([torch_weights[-1].t().detach().numpy(), torch_bias[-1].detach().numpy()])

keras_model.model.save(args.output)

print(torch_model)
keras_model.model.summary()

if not args.flatten:
    pred_data = pred_data.reshape(pred_data.shape[0], -1)

tf_predict = keras_model.model.predict(pred_data)
torch_predict = torch_model(torch.Tensor(pred_data)).detach().numpy()
print('prediction difference:', np.linalg.norm((tf_predict - torch_predict).flatten(), ord=1) / n_data)
true_labels = np.argmax(pred_label, axis=1)
tf_labels = np.argmax(tf_predict, axis=1)
torch_labels = np.argmax(torch_predict, axis=1)
print('tensorflow acc:', np.sum(true_labels == tf_labels) / n_data)
print('pytorch acc:', np.sum(true_labels == torch_labels) / n_data)

