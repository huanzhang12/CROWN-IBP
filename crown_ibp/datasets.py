## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import multiprocessing
import torch
from torch.utils import data
from functools import partial
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# compute image statistics (by Andreas https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/4)
def get_stats(loader):
    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0) 
        reshaped_img = images.view(batch_samples, images.size(1), -1)
        mean += reshaped_img.mean(2).sum(0)
    w = images.size(2)
    h = images.size(3)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*w*h))
    return mean, std

# load MNIST of Fashion-MNIST
def mnist_loaders(dataset, batch_size, shuffle_train = True, shuffle_test = False, normalize_input = False, num_examples = None, test_batch_size=None): 
    mnist_train = dataset("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = dataset("./data", train=False, download=True, transform=transforms.ToTensor())
    if num_examples:
        indices = list(range(num_examples))
        mnist_train = data.Subset(mnist_train, indices)
        mnist_test = data.Subset(mnist_test, indices)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=shuffle_train, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),2))
    if test_batch_size:
        batch_size = test_batch_size
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),2))
    std = [1.0]
    mean = [0.0]
    train_loader.std = std
    test_loader.std = std
    train_loader.mean = mean
    test_loader.mean = mean
    return train_loader, test_loader

def cifar_loaders(batch_size, shuffle_train = True, shuffle_test = False, train_random_transform = False, normalize_input = False, num_examples = None, test_batch_size=None): 
    if normalize_input:
        std = [0.2023, 0.1994, 0.2010]
        mean = [0.4914, 0.4822, 0.4465]
        normalize = transforms.Normalize(mean = mean, std = std)
    else:
        std = [1.0, 1.0, 1.0]
        mean = [0, 0, 0]
        normalize = transforms.Normalize(mean = mean, std = std)
    if train_random_transform:
        if normalize_input:
            train = datasets.CIFAR10('./data', train=True, download=True, 
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            train = datasets.CIFAR10('./data', train=True, download=True, 
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                ]))
    else:
        train = datasets.CIFAR10('./data', train=True, download=True, 
            transform=transforms.Compose([transforms.ToTensor(),normalize]))
    test = datasets.CIFAR10('./data', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    if num_examples:
        indices = list(range(num_examples))
        train = data.Subset(train, indices)
        test = data.Subset(test, indices)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=shuffle_train, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),6))
    if test_batch_size:
        batch_size = test_batch_size
    test_loader = torch.utils.data.DataLoader(test, batch_size=max(batch_size, 1),
        shuffle=shuffle_test, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),6))
    train_loader.std = std
    test_loader.std = std
    train_loader.mean = mean
    test_loader.mean = mean
    return train_loader, test_loader

def svhn_loaders(batch_size, shuffle_train = True, shuffle_test = False, train_random_transform = False, normalize_input = False, num_examples = None, test_batch_size=None): 
    if normalize_input:
        mean = [0.43768206, 0.44376972, 0.47280434] 
        std = [0.19803014, 0.20101564, 0.19703615]
        normalize = transforms.Normalize(mean = mean, std = std)
    else:
        std = [1.0, 1.0, 1.0]
        mean = [0, 0, 0]
        normalize = transforms.Normalize(mean = mean, std = std)
    if train_random_transform:
        if normalize_input:
            train = datasets.SVHN('./data', split='train', download=True, 
                transform=transforms.Compose([
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            train = datasets.SVHN('./data', split='train', download=True, 
                transform=transforms.Compose([
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                ]))
    else:
        train = datasets.SVHN('./data', split='train', download=True, 
            transform=transforms.Compose([transforms.ToTensor(),normalize]))
    test = datasets.SVHN('./data', split='test', download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    if num_examples:
        indices = list(range(num_examples))
        train = data.Subset(train, indices)
        test = data.Subset(test, indices)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=shuffle_train, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),6))
    if test_batch_size:
        batch_size = test_batch_size
    test_loader = torch.utils.data.DataLoader(test, batch_size=max(batch_size, 1),
        shuffle=shuffle_test, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),6))
    train_loader.std = std
    test_loader.std = std
    train_loader.mean = mean
    test_loader.mean = mean
    mean, std = get_stats(train_loader)
    print('dataset mean = ', mean.numpy(), 'std = ', std.numpy())
    return train_loader, test_loader

# when new loaders is added, they must be registered here
loaders = {
        "mnist": partial(mnist_loaders, datasets.MNIST),
        "fashion-mnist": partial(mnist_loaders, datasets.FashionMNIST),
        "cifar": cifar_loaders,
        "svhn": svhn_loaders,
        }

