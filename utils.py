import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import math


def patchswap(args, x1, y1, alpha=1.0):
    m = x1.shape[0]

    new_idx = torch.randperm(m)
    x2 = x1[new_idx]
    y2 = y1[new_idx]

    x1 = x1.unfold(2, args.patch_size, args.patch_size).unfold(3, args.patch_size, args.patch_size)
    x1 = x1.reshape(-1, args.num_channels, args.num_patches, args.patch_size, args.patch_size)

    x2 = x2.unfold(2, args.patch_size, args.patch_size).unfold(3, args.patch_size, args.patch_size)
    x2 = x2.reshape(-1, args.num_channels, args.num_patches, args.patch_size, args.patch_size)

    lam = np.random.beta(alpha, alpha)

    x1_frames_count = round(lam * args.num_patches)
    lam_new = x1_frames_count / args.num_patches

    rand_probs = torch.rand(m, args.num_patches).cuda()

    if x1_frames_count > 0:
        thresh = -torch.kthvalue(-rand_probs, x1_frames_count, -1, keepdims=True)[0]
    else:
        thresh = torch.Tensor([1.1]).cuda()

    swap = rand_probs >= thresh
    swap = swap.reshape(m, 1, args.num_patches, 1, 1)
    x = x1 * swap + x2 * (~swap)

    x = x.contiguous().view(m, args.num_channels, -1, args.patch_size * args.patch_size)
    x = x.permute(0, 1, 3, 2)
    x = x.contiguous().view(m, args.num_channels * args.patch_size * args.patch_size, -1)
    x = F.fold(x, output_size=(args.img_size, args.img_size), kernel_size=args.patch_size, stride=args.patch_size)

    return x, y1, y2, lam_new


def patchswap_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b) + lam * np.log(lam + 1e-6) + (1 - lam) * np.log(1 - lam + 1e-6)


def accuracy(args, target, output, k=5):
    top1 = top_k_accuracy_score(target, output, k=1, labels=range(0, args.num_classes))
    topk = top_k_accuracy_score(target, output, k=k, labels=range(0, args.num_classes))
    return top1, topk


def cmatrix(args, target, predict):
    return confusion_matrix(target, np.array(predict).argmax(1), labels=range(0, args.num_classes))


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'Epoch: {epoch + 1}\tlr: {lr}')
