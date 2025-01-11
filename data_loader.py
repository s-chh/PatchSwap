import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def get_data_loader(args):
	train_transform = []
	if args.randomresizecrop:
		train_transform.append(transforms.RandomResizedCrop(args.image_size))
	elif args.padding > 0:
		train_transform.append(transforms.Resize(args.image_size))
		train_transform.append(transforms.RandomCrop([args.image_size, args.image_size], padding=args.padding))
	elif args.resizecrop > 0:
		train_transform.append(transforms.Resize(args.resizecrop))
		train_transform.append(transforms.RandomCrop(args.image_size))
	else:
		train_transform.append(transforms.Resize([args.image_size, args.image_size]))

	if args.hflip:
		train_transform.append(transforms.RandomHorizontalFlip())

	train_transform.append(transforms.ToTensor())
	train_transform.append(transforms.Normalize(args.mean, args.std))
	train_transform = transforms.Compose(train_transform)

	test_transform = []
	if args.image_size == 224:
		test_transform.append(transforms.Resize(256))
		test_transform.append(transforms.CenterCrop(224))
	else:
		test_transform.append(transforms.Resize([args.image_size, args.image_size]))

	test_transform.append(transforms.ToTensor())
	test_transform.append(transforms.Normalize(args.mean, args.std))
	test_transform = transforms.Compose(test_transform)

	if args.dataset.lower() == 'cifar10':
		train = datasets.CIFAR10(root=args.data_path, train=True, transform=train_transform, download=True)
		test = datasets.CIFAR10(root=args.data_path, train=False, transform=test_transform, download=True)
	elif args.dataset.lower() == 'cifar100':
		train = datasets.CIFAR100(root=args.data_path, train=True, transform=train_transform, download=True)
		test = datasets.CIFAR100(root=args.data_path, train=False, transform=test_transform, download=True)
	elif args.dataset.lower() == 'fmnist':
		train = datasets.FashionMNIST(root=args.data_path, train=True, transform=train_transform, download=True)
		test = datasets.FashionMNIST(root=args.data_path, train=False, transform=test_transform, download=True)
	elif args.dataset.lower() == 'svhn':
		train = datasets.SVHN(root=args.data_path, split='train', transform=train_transform, download=True)
		test = datasets.SVHN(root=args.data_path, split='test', transform=test_transform, download=True)
	else:
		train = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=train_transform)
		test = datasets.ImageFolder(root=os.path.join(args.data_path, 'test'), transform=test_transform)

	train_loader = DataLoader(dataset=train,
							batch_size=args.batch_size,
							shuffle=True,
							num_workers= args.n_workers,
							drop_last=True)

	test_loader = DataLoader(dataset=test,
							batch_size=args.batch_size,
							shuffle=True,
							num_workers=args.n_workers,
							drop_last=False)

	return train_loader, test_loader
