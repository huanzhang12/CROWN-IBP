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

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',help='input file', required=True)
parser.add_argument('-o','--output',help='output file', required=True)
args = parser.parse_args()

data = MNIST()
n_data = 500
pred_data = data.test_data[:n_data]
pred_label = data.test_labels[:n_data]

weight_dims, activation, activation_param, _ = get_model_meta(args.input)
keras_model = NLayerModel(weight_dims[:-1], args.input, activation=activation, activation_param=activation_param)

modules = []

class Flatten(nn.Module):
    def forward(self, x):
        y = x.view(x.size(0), -1)
        return y

for l in keras_model.model.layers:
    if isinstance(l, TFDense) or isinstance(l, KerasDense):
        linear = nn.Linear(l.input_shape[1], l.output_shape[1])
        w, b = l.get_weights()
        linear.weight.data.copy_(torch.Tensor(w.T.copy()))
        linear.bias.data.copy_(torch.Tensor(b))
        modules.append(linear)
    elif isinstance(l, TFFlatten) or isinstance(l, KerasFlatten):
        modules.append(Flatten())
    elif isinstance(l, TFActivation) or isinstance(l, KerasActivation):
        if 'relu' in str(l.activation):
            modules.append(nn.ReLU())
        else:
            raise(ValueError("Unsupported acitation"))
    else:
        print(l)
        raise(ValueError("Unknow layer", l))

torch_model = nn.Sequential(*modules)
torch.save(torch_model.state_dict(), args.output)

print(torch_model)

tf_predict = keras_model.model.predict(pred_data)
torch_predict = torch_model(torch.Tensor(pred_data)).detach().numpy()
print('prediction difference:', np.linalg.norm((tf_predict - torch_predict).flatten(), ord=1) / n_data)
true_labels = np.argmax(pred_label, axis=1)
tf_labels = np.argmax(tf_predict, axis=1)
torch_labels = np.argmax(torch_predict, axis=1)
print('tensorflow acc:', np.sum(true_labels == tf_labels) / n_data)
print('pytorch acc:', np.sum(true_labels == torch_labels) / n_data)

