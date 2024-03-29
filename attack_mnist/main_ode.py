import os
from os import sys
import random
from copy import deepcopy

import time
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append("../ad_attack")
sys.path.append("../mnist")
from adversarial_attack_method import adversarial_attack

from options import args
from MiscTools import get_logger, makedirs, add_noise_tensor, count_parameters
from networks import ODENet_MNIST

if args.isRandom == False:
    random.seed(0)
    np.random.seed(seed=0)
    torch.manual_seed(0)

def add_noise_tensor_random(x):
    isAdd = random.choice([True])
    noise_level = random.choice([100])
    if isAdd:
        return add_noise_tensor(x, ['G', noise_level])
    else:
        return x

def get_mnist_loaders(isTrain=False, batch_size=128, test_batch_size=1000):
    if isTrain:
        transform = transforms.Compose([
            transforms.ToTensor()
            ])
        train_loader = DataLoader(
            datasets.MNIST(root='../../data/mnist', train=True, download=True, transform=transform), 
            batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        train_eval_loader = DataLoader(
            datasets.MNIST(root='../../data/mnist', train=True, download=True, transform=transform),
            batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
        )
        test_loader = DataLoader(
            datasets.MNIST(root='../../data/mnist', train=False, download=True, transform=transform),
            batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
        )

        transform_n = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: add_noise_tensor_random(x))
                ])
        test_loader_n = DataLoader(
            datasets.MNIST(root='../../data/mnist', train=False, download=True, transform=transform_n),
            batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True)

        return train_loader, test_loader, test_loader_n, train_eval_loader
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
            ])
        transform_noisy = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: add_noise_tensor(x, ['G', args.noise_level]))
            ])
        test_loader = DataLoader(
            datasets.MNIST(root='../../data/mnist', train=False, download=True, transform=transform),
            batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
        )
        test_loader_noisy = DataLoader(
            datasets.MNIST(root='../../data/mnist', train=False, download=True, transform=transform_noisy),
            batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
        )
    return test_loader, test_loader_noisy 


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader):
    
    total_correct = 0
    for x, y in dataset_loader:
        x = x.cuda()
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def accuracy_withRef(model, RefDL, PertbDL):
    target_class = np.array([])
    pred_class_ref = np.array([])
    pred_class_pertb = np.array([])
    for x, y in RefDL:
        x = x.cuda(args.device_ids[0])
        pred_class_ref = np.concatenate((pred_class_ref, np.argmax(model(x).cpu().detach().numpy(), axis=1)), axis=None)
        # pred_class_ref.append(np.argmax(model(x).cpu().detach().numpy(), axis=1))

    for x, y in PertbDL:
        x = x.cuda(args.device_ids[0])
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.concatenate((target_class, np.argmax(y, axis=1)), axis=None)
        pred_class_pertb = np.concatenate((pred_class_pertb, np.argmax(model(x).cpu().detach().numpy(), axis=1)), axis=None)

    accu_ref =  np.sum(pred_class_ref == target_class) / len(target_class)
    accu_pertb_target = np.sum(pred_class_pertb == target_class)/len(target_class)
    accu_pertb_ref = np.sum((pred_class_ref == target_class) & (pred_class_pertb == target_class)) / np.sum(pred_class_ref == target_class)
    return accu_ref, accu_pertb_target, accu_pertb_ref


if __name__ == '__main__':
    # Build model
    print('===> Building model ...')
    # Add logs
    save_dir = os.path.join(args.dir_logging); makedirs(save_dir)
    logger = get_logger(logpath=os.path.join(save_dir, args.logging_file))
    logger.info(os.path.abspath(__file__))
    # for arg in vars(args):
    #     logger.info('{}: {}'.format(arg, getattr(args, arg)))
    logger.info('{}: {}'.format('dir_model', getattr(args, 'dir_model')))
    logger.info('{}: {}'.format('step_size', getattr(args, 'step_size')))
    # build model
    model = ODENet_MNIST()
    if torch.cuda.is_available():
        model = model.cuda(args.device_ids[0])
    if len(args.device_ids) >= 2:
        model = nn.DataParallel(model, args.device_ids)
    
    model.load_state_dict(torch.load(args.dir_model).state_dict())

    # Construct datasets
    test_loader, _ = get_mnist_loaders(
        isTrain=False, batch_size=128, test_batch_size=1)
    # Testing
    model.eval()
    # epsilon, learning rate
    model.eval()
    fgsm1 = 0 if False else adversarial_attack(1000,model,test_loader,0.15,0.15,1)
    fgsm3 = 0 if False else adversarial_attack(1000,model,test_loader,0.3,0.3,1)
    fgsm5 = 0 if False else adversarial_attack(1000,model,test_loader,0.5,0.5,1)
    pgd2 = 0 if True else adversarial_attack(1000,model,test_loader,0.2,0.04,10)
    pgd3 = 0 if True else adversarial_attack(1000,model,test_loader,0.3,0.06,10)
    black = 0 if True else adversarial_attack(1000,model,test_loader,0.4,0.05,35,isblackbox=True)
    logger.info(
            "FGSM 0.15: {} FGSM 0.3: {}, FGSM 0.5: {} PGD 0.2: {}  PGD 0.3 :{} black:{}".format(fgsm1,fgsm3,fgsm5,pgd2,pgd3,black)
        )
      