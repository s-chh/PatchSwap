import torch
import torch.nn as nn
import os
import numpy as np
from torch import optim
from sklearn.manifold import TSNE
from data_loader import get_data_loader
from model import ViT
from utils import patchswap, patchswap_loss, accuracies, cmatrix, plot_tsne


class Solver(object):
    def __init__(self, args):
        self.args = args

        self.train_loader, self.test_loader = get_data_loader(self.args)

        self.net = ViT(self.args.image_size, self.args.patch_size, self.args.n_channels, self.args.n_classes).cuda()

        print("Network:")
        print(self.net)

        self.ce_loss = nn.CrossEntropyLoss()

    def train(self):
        iter_per_epoch = len(self.train_loader)

        print(f"Iters per epoch: {iter_per_epoch:d}\n")

        optimizer = optim.AdamW(self.net.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.epochs - self.args.warmup, eta_min=1e-5)
        linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/self.args.warmup, end_factor=1.0, total_iters=self.args.warmup-1, last_epoch=-1)

        best_acc = 0
        for epoch in range(self.args.epochs):
            print(f"Ep:[{epoch + 1}/{self.args.epochs}] lr:{optimizer.param_groups[0]['lr']:.6f}")

            self.net.train()
            for i, data in enumerate(self.train_loader):
                x, y = data
                x, y = x.cuda(), y.cuda()

                if self.args.method == 'patchswap':
                    x, y1, y2, lam_ = patchswap(x, y, self.args.patch_size)
                    logits = self.net(x)
                    loss = patchswap_loss(self.ce_loss, logits, y1, y2, lam_)

                else:
                    logits = self.net(x)
                    loss = self.ce_loss(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 50 == 0:
                    print(f"Ep:[{epoch + 1}/{self.args.epochs}] It: {i + 1}/{iter_per_epoch}, loss:{loss:.4f}")

            torch.save(self.net.state_dict(), os.path.join(self.args.model_path, f"{self.args.method}.pt"))
            test_acc = self.test(train=False)
            best_acc = max(test_acc, best_acc)
            print(f"Best test Top-1 acc: {best_acc:.2%}\n")

            if epoch < self.args.warmup:
                linear_warmup.step()
            else:
                cos_decay.step()

    def compute_test_metric(self, loader):
        self.net.eval()

        actual = []
        predictions = []

        for data in loader:
            x, y = data
            x = x.cuda()

            with torch.no_grad():
                logits = self.net(x)

            actual.append(y)
            predictions.append(logits.cpu())

        actual = torch.cat(actual)
        predictions = torch.cat(predictions)

        acc1, acc5 = accuracies(actual, predictions, self.args.n_classes, 5)
        loss = self.ce_loss(predictions, actual)
        cm = cmatrix(actual, predictions, self.args.n_classes)

        return acc1, acc5, loss, cm

    def test(self, train=False):

        if train:
            acc1, acc5, loss, cm = self.compute_test_metric(self.train_loader)
            print(f"Train Accuracy Top-1: {acc1:.2%} Top-5: {acc5:.2%}\tLoss: {loss:.2f}")
            if self.args.cm:
                print(cm)

        acc1, acc5, loss, cm = self.compute_test_metric(self.test_loader)
        print(f"Test Accuracy Top-1: {acc1:.2%} Top-5: {acc5:.2%}\tLoss: {loss:.2f}")
        if self.args.cm:
            print(cm)
        return acc1

    def deep_features(self, loader, samples_to_use=500):
        features = []
        labels = []
        count = 0

        self.net.eval()
        for i, data in enumerate(loader):
            if count > samples_to_use and samples_to_use > 0:
                break
            x, y = data

            with torch.no_grad():
                _, feature = self.net(x.cuda(), deep=True)

            features.append(feature.detach().cpu().numpy())
            labels.append(y)
            count = count + feature.shape[0]

        features = np.vstack(features)
        labels = np.hstack(labels).reshape(-1)

        return features, labels

    def tsne(self):
        x, y = self.deep_features(self.train_loader, samples_to_use=50 * self.args.n_classes)
        print("on tsne")
        x = TSNE().fit_transform(x)
        if self.args.dataset == 'cifar10':
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.args.dataset == 'fmnist':
            class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        else:
            class_names = list(range(self.args.n_classes))
        plot_tsne(x, y, os.path.join(self.args.tsne_path, f"{self.args.method}.png"), class_names, legends=self.args.cm)

