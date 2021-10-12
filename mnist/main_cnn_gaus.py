import os
from os import sys
import random

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

from options import args
from MiscTools import get_logger, makedirs, add_noise_tensor, count_parameters
from networks import CNN_MNIST

if args.isRandom == False:
    random.seed(0)
    np.random.seed(seed=0)
    torch.manual_seed(0)

def add_noise_tensor_random(x):
    isAdd = random.choice([True, False])
    noise_level = random.choice([50, 75, 100])
    # isAdd = True
    # noise_level = 75
    if isAdd:
        return add_noise_tensor(x, ['G', noise_level])
    else:
        return x


def get_mnist_loaders(isTrain=False, batch_size=128, test_batch_size=1000):
    if isTrain:
        # train
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: add_noise_tensor_random(x))
                ])
        train_loader = DataLoader(
            datasets.MNIST(root='../../data/mnist', train=True, download=True, transform=transform), 
            batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
        train_eval_loader = DataLoader(
            datasets.MNIST(root='../../data/mnist', train=True, download=True, transform=transform),
            batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True)
        test_loader = DataLoader(
            datasets.MNIST(root='../../data/mnist', train=False, download=True, transform=transform),
            batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True)

        return train_loader, test_loader, train_eval_loader
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
        x = x.cuda(args.device_ids[0])
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
    if args.isTrain:
        # Add logs
        save_dir = os.path.join(args.dir_logging, args.exp_name); makedirs(save_dir)
        logger = get_logger(logpath=os.path.join(save_dir,'train_result.txt'))   
        logger.info(os.path.abspath(__file__))
        for arg in vars(args):
            logger.info('{}: {}'.format(arg, getattr(args, arg)))

        # Build model
        model = CNN_MNIST()
        criterion = nn.CrossEntropyLoss() 
        if torch.cuda.is_available():
            model = model.cuda(args.device_ids[0])
            criterion = criterion.cuda(args.device_ids[0])
        if len(args.device_ids) >= 2:
            model = nn.DataParallel(model, args.device_ids)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        if not args.resume:
            logger.info('---- Model ----')
            logger.info(model)
            logger.info('Number of parameters: {}'.format(count_parameters(model))) 

        # Construct datasets
        train_loader, test_loader, train_eval_loader = get_mnist_loaders(
            isTrain=True, batch_size=128, test_batch_size=1000)

        # Training
        logger.info('---- Training ----')
        best_epoch = {'epoch':0, 'acc':0}

            # Resume model
        if args.resume:
            print("=> loading checkpoint '{}'".format(save_dir))
            checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
            args.start_epoch = checkpoint['epoch']+1
            best_epoch = checkpoint['best_epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(save_dir, checkpoint['epoch']))

        for epoch in range(args.start_epoch, args.end_epoch):
            tic = time.time()
            scheduler.step(epoch)
            # training
            for _, batch_tr in enumerate(train_loader):
                optimizer.zero_grad()
                loss = criterion(model(batch_tr[0].cuda(args.device_ids[0])), batch_tr[1].cuda(args.device_ids[0]))
                loss.backward()
                optimizer.step()
            # evaluation
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                test_acc = accuracy(model, test_loader)
            # logging
            if test_acc >= best_epoch['acc']:
                best_epoch['epoch'] = epoch
                best_epoch['acc'] = test_acc
                torch.save(model, os.path.join(save_dir, 'model_best.pth'))
            
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_epoch': best_epoch,
                'optimizer': optimizer.state_dict()
                }
            torch.save(checkpoint,os.path.join(save_dir, 'checkpoint.pth'))

            logger.info(
                "Epoch {:04d}/{:04d} | Time {:.3f}s | "
                "Train Acc {:.4f} | Test Acc {:.4f} | Best epoch @ {:04d} with Acc {:.4f} | lr: {:.6f}".format(
                    epoch, args.end_epoch, time.time()-tic, train_acc, test_acc, best_epoch['epoch'], 
                    best_epoch['acc'], optimizer.state_dict()['param_groups'][0]['lr'])
                )
    else:
        # Add logs
        save_dir = os.path.join(args.dir_logging); makedirs(save_dir)
        logger = get_logger(logpath=os.path.join(save_dir, args.logging_file))
        logger.info(os.path.abspath(__file__))
        # for arg in vars(args):
        #     logger.info('{}: {}'.format(arg, getattr(args, arg)))
        logger.info('{}: {}'.format('dir_model', getattr(args, 'dir_model')))
        logger.info('{}: {}'.format('noise_level', getattr(args, 'noise_level')))
        # Build model
        print('===> Building model ...')
        model = CNN_MNIST()
        if torch.cuda.is_available():
            model = model.cuda(args.device_ids[0])
        if len(args.device_ids) >= 2:
            model = nn.DataParallel(model, args.device_ids)
        model.load_state_dict(torch.load(args.dir_model).state_dict())
        # Construct datasets
        test_loader, test_loader_noisy = get_mnist_loaders(
            isTrain=False, batch_size=128, test_batch_size=1000)
        # Testing
        model.eval()
        with torch.no_grad():
            accu_orgin, accu_pertb, accu_pertb_orgin = accuracy_withRef(model, test_loader, test_loader_noisy)
        logger.info(
            "Test Acc: {}, Pertb Acc: {}, Pert_wrt_orgin: {}".format(accu_orgin, accu_pertb, accu_pertb_orgin)
            )