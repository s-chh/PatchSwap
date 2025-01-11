import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, top_k_accuracy_score


def patchify(x, patch_size):
    """
        function for patchify an input image.
        # B, C, H, W ; P  -->  B, -1, P*P*C
    """
    n_patches = (x.shape[-1] // patch_size) ** 2                                    # (H//P) ^ 2
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)       # B, C, H, W             -->  B, C, H//P, W//P, P, P
    x = x.permute(0, 2, 3, 1, 4, 5)                                                 # B, C, H//P, W//P, P, P --> B, H//P, W//P, C, P, P
    x = x.reshape(x.shape[0], n_patches, -1)                                        # B, H//P, W//P, C, P, P --> B, (H//P* W//P), (C*P*P)
    return x


def unpatchify(x, patch_size):
    """
        function for unpatchify a sequence to an image.
        # B, S, E; P   -->  B, C, H, W
        where  S = (H//P* W//P)  and  E = P*P*C
    """
    h = int(x.shape[1] ** 0.5 * patch_size)                                         # H = (S^0.5) * P
    x = x.permute(0, 2, 1)                                                          # B, S, E  -->  B, E, S
    x = F.fold(x, output_size=(h, h), kernel_size=patch_size, stride=patch_size)    # B, E, S  -->  B, C, H, W
    return x


def patchswap(x1, y1, patch_size, alpha=1.0):
    """
        patch swap function.

        Inputs:
            x1: image tensor
            y1: image label
            patch_size: patch size
            alpha: alpha hyperparameter used for generating lambda

        Outputs:
            x: PatchSwap image tensor
            y1: label of the first image used in PatchSwap
            y2: label of the second image used in PatchSwap
            lam_: lambda value used for generating PatchSwap images.
    """

    x1 = patchify(x1, patch_size)                                                   # B, C, H, W  -->  B, -1, P*P*C  Image to Patches

    m, n_patches, e = x1.shape                                                       # B, S, E

    # Randomly ordering for combining
    new_idx = torch.randperm(m)
    x2 = x1[new_idx]
    y2 = y1[new_idx]

    lam = np.random.beta(alpha, alpha)                                              # Lambda hyper-parameter
    x1_frames_count = round(lam * n_patches)                                        # Rounded number of frames to swap
    lam_ = x1_frames_count / n_patches                                              # Updated lambda

    # Generate different mask for each pair using random noise while keeping counts the same.
    mask = torch.zeros_like(x1[:, :, 0])
    mask[:, :x1_frames_count] = 1
    rand_noise = torch.rand_like(mask)
    noise_sort = rand_noise.argsort(1)
    mask = torch.gather(mask, dim=1, index=noise_sort)
    mask = mask.unsqueeze(-1).bool()

    x = x1 * mask + x2 * (~mask)                                                    # Combining the images
    x = unpatchify(x, patch_size)                                                   # Unpatchify sequence back to image

    return x, y1, y2, lam_


def patchswap_loss(criterion, pred, y_a, y_b, lam):
    """
        patch swap loss function.

        Inputs:
            criterion: loss function to be used
            pred     : logits
            y_a      : label of the first image used in PatchSwap
            y_b      : label of the second image used in PatchSwap
            lam      : lambda value used for generating PatchSwap images

        Outputs:
           PatchSwap loss value
    """
    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)                    # loss
    normalized_loss = loss + lam * np.log(lam + 1e-6) + (1 - lam) * np.log(1 - lam + 1e-6)  # Normalized to be lowest at 0.
    return normalized_loss


def accuracies(target, output, n_classes, k=5):
    """
        function for calculating accuracies.

        Inputs:
            target    : labels
            output    : probabilites/logits
            n_classes : total number of classes
            k         : k for calculating top-k accuracy

        Outputs:
           top1 and topk accuracy
    """
    top1 = top_k_accuracy_score(target, output, k=1, labels=range(n_classes))
    topk = top_k_accuracy_score(target, output, k=k, labels=range(n_classes))
    return top1, topk


def cmatrix(target, predict, n_classes):
    """
        function for calculating confusion matrix.

        Inputs:
            target    : labels
            output    : probabilites/logits
            n_classes : total number of classes

        Outputs:
           confusion matrix
    """
    return confusion_matrix(target, np.array(predict).argmax(1), labels=range(n_classes))


def plot_tsne(x, y, fname, class_names=[], legends=False):  # Plotting the graph
    num_classes = y.max() + 1

    palettes = np.array(sns.color_palette('deep', n_colors=num_classes))

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    for i in range(num_classes):
        samples = x[y == i]
        plt.scatter(samples[:, 0], samples[:, 1], label=class_names[i], color=palettes[i], marker='o', s=10)

    ax.axis('off')
    ax.axis('tight')
    if legends:
        plt.legend()
    plt.savefig(fname)
    plt.close('all')
    return
