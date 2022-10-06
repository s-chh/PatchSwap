import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch


def get_loaders(args):
    train_augmentations = [transforms.Resize(args.img_size)]
    if args.padding > 0:
        train_augmentations += [transforms.RandomCrop(args.img_size, padding=args.padding)]
    if args.hflip:
        train_augmentations += [transforms.RandomHorizontalFlip()]
    train_augmentations += [transforms.ToTensor(), transforms.Normalize(mean=args.mean, std=args.std)]
    train_augmentations = transforms.Compose(train_augmentations)

    if args.dataset.lower().startswith('cifar'):
        train_dataset = datasets.__dict__[args.dataset.upper()](root=os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_augmentations)
    elif args.dataset == 'fmnist':
        train_dataset = datasets.FashionMNIST(root=os.path.join(args.data_path, args.dataset), train=True, download=True, transform=train_augmentations)
    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(root=os.path.join(args.data_path, args.dataset), split='train', download=True, transform=train_augmentations)
    else:
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, args.dataset, 'train'), transform=train_augmentations)

    test_augmentations = transforms.Compose([transforms.Resize(args.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=args.mean, std=args.std)])

    if args.dataset.lower().startswith('cifar'):
        val_dataset = datasets.__dict__[args.dataset.upper()](root=os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_augmentations)
    elif args.dataset == 'fmnist':
        val_dataset = datasets.FashionMNIST(root=os.path.join(args.data_path, args.dataset), train=False, download=True, transform=test_augmentations)
    elif args.dataset == 'svhn':
        val_dataset = datasets.SVHN(root=os.path.join(args.data_path, args.dataset), split='test', download=True, transform=test_augmentations)
    else:
        val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, args.dataset, 'val'), transform=test_augmentations)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             drop_last=False)
    return train_loader, val_loader
