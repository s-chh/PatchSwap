# ## CIFAR 10 ## 
python -u main.py --dataset cifar10 --patch_size 4
python -u main.py --dataset cifar10 --patch_size 8
python -u main.py --dataset cifar10 --patch_size 16

# ## CIFAR 100 ## 
python -u main.py --dataset cifar100 --patch_size 4
python -u main.py --dataset cifar100 --patch_size 8
python -u main.py --dataset cifar100 --patch_size 16

# ## FashionMNIST ## 
python -u main.py --dataset fmnist --patch_size 4
python -u main.py --dataset fmnist --patch_size 8
python -u main.py --dataset fmnist --patch_size 16

# ## SVHN ## 
python -u main.py --dataset svhn --patch_size 4
python -u main.py --dataset svhn --patch_size 8
python -u main.py --dataset svhn --patch_size 16

# ## TinyImageNet ## 
python -u main.py --dataset ti --patch_size 8 --data_path /path
python -u main.py --dataset ti --patch_size 16 --data_path /path

