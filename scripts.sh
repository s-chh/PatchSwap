## CIFAR 10 ## 
python -u main.py --dataset cifar10 --patch_size 4 &> c10_4.txt
python -u main.py --dataset cifar10 --patch_size 8 &> c10_8.txt
python -u main.py --dataset cifar10 --patch_size 16 &> c10_16.txt

## CIFAR 100 ## 
python -u main.py --dataset cifar100 --patch_size 4 &> c100_4.txt
python -u main.py --dataset cifar100 --patch_size 8 &> c100_8.txt
python -u main.py --dataset cifar100 --patch_size 16 &> c100_16.txt

## FashionMNIST ## 
python -u main.py --dataset fmnist --patch_size 4 &> fmnist_4.txt
python -u main.py --dataset fmnist --patch_size 8 &> fmnist_8.txt
python -u main.py --dataset fmnist --patch_size 16 &> fmnist_16.txt

## SVHN ## 
python -u main.py --dataset svhn --patch_size 4 &> svhn_4.txt
python -u main.py --dataset svhn --patch_size 8 &> svhn_8.txt
python -u main.py --dataset svhn --patch_size 16 &> svhn_16.txt

## TinyImageNet ## 
python -u main.py --dataset ti --patch_size 8 --data_path <datasetpath> &> ti_8.txt
python -u main.py --dataset ti --patch_size 16 --data_path <datasetpath> &> ti_16.txt

