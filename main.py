import argparse
from time import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import json
from utils import patchswap, patchswap_loss, accuracy, adjust_learning_rate, cmatrix
import os
import numpy as np
from data_loader import get_loaders
from model import ViT


def init_parser():
    parser = argparse.ArgumentParser(description='PatchSwap')
    parser.add_argument('--data_path', metavar='DIR', default='./data/', help='path to dataset')
    parser.add_argument('--dataset', type=str.lower, choices=['cifar10', 'cifar100', 'fmnist', 'svhn', 'ti'], default='cifar10')
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--model_path', type=str, default='./model/')
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=3e-2)
    return parser


def update_args(args):
    with open(os.path.join("config",args.dataset+".json")) as data_file:
        config = json.load(data_file)

    args.img_size = config["img_size"]
    args.hflip = config["hflip"]
    args.num_channels = config["num_channels"]
    args.num_classes = config["num_classes"]
    args.cm = config["cm"]
    args.padding = config["padding"]
    args.mean = config["mean"]
    args.std = config["std"]

    args.is_cuda = torch.cuda.is_available()
    args.num_patches = (args.img_size//args.patch_size) ** 2
    args.model_path = os.path.join(args.model_path, args.dataset)
    os.makedirs(args.model_path, exist_ok=True)

    return args


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


def main():
    parser = init_parser()
    args = parser.parse_args()
    args = update_args(args)
    print_args(args)

    train_loader, val_loader = get_loaders(args)

    model = ViT(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()
    criterion_ = nn.CrossEntropyLoss(reduction='none')

    if args.is_cuda:
        model = model.cuda()

    print("\nBeginning training")
    time_begin = time()

    best_acc1 = 0
    best_acc5 = 0
    te_acc1 = 0
    te_acc5 = 0

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        cls_train(args, train_loader, model, criterion, optimizer)

        if (epoch + 1) % 25 == 0:
            train_acc1, train_acc5, train_cm, train_loss = cls_validate(args, train_loader, model, criterion_)
            print(f'[Epoch: {epoch + 1}]\tTrain Top1: {train_acc1:.1f}%\tTop5: {train_acc5:.1f}%\tLoss: {train_loss:.4f}')
            if args.cm:
                print(train_cm)

        te_acc1, te_acc5, te_cm, te_loss = cls_validate(args, val_loader, model, criterion_)

        best_acc1 = max(te_acc1, best_acc1)
        best_acc5 = max(te_acc5, best_acc5)

        total_mins = -1 if time_begin is None else (time() - time_begin) / 60
        print(f'[Epoch: {epoch + 1}]\tTest Top1: {te_acc1:.1%}\tTop5: {te_acc5:.1%}\tLoss: {te_loss:.4f}')
        if args.cm:
            print(te_cm)
        print(f'Best Top1: {best_acc1:.1%}\tTop5: {best_acc5:.1%}\tTime: {total_mins:.2f}')

        print()

        torch.save(model.state_dict(), os.path.join(args.model_path, 'model.pt'))

    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'best top-1: {best_acc1:.2f}, '
          f'best top-5: {best_acc5:.2f}, '
          f'final top-1: {te_acc1:.2f}, '
          f'final top-5: {te_acc5:.2f}')


def cls_train(args, train_loader, model, criterion, optimizer):

    model.train()
    iter_per_epoch = len(train_loader)
    for i, (x, y) in enumerate(train_loader):
        if args.is_cuda:
            x = x.cuda()
            y = y.cuda()

        x_mix, y1, y2, lam = patchswap(args, x, y)
        output = model(x_mix)
        loss = patchswap_loss(criterion, output, y1, y2, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0 or i == iter_per_epoch-1:
            loss = loss.item() + lam * np.log(lam+1e-6) + (1-lam) * np.log(1-lam+1e-6)
            print(f'It: {i+1}/{iter_per_epoch}\tLoss: {loss:.4f}')


def cls_validate(args, loader, model, criterion):
    model.eval()
    actual, predict, losses = [], [], []

    with torch.no_grad():
        for (x, y) in loader:
            if args.is_cuda:
                x = x.cuda()

            output = model(x)

            losses += criterion(output, y.cuda()).tolist()
            actual += y.tolist()
            predict += output.tolist()
    acc1, acc5 = accuracy(args, actual, predict)
    avg_loss = sum(losses)/len(losses)
    cm = cmatrix(args, actual, predict)

    return acc1, acc5, cm, avg_loss


if __name__ == '__main__':
    main()
