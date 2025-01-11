import os
import json
import datetime
import argparse
from solver import Solver


def main(args):
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.tsne_path, exist_ok=True)

    solver = Solver(args)

    solver.train()
    solver.test(train=True)
    solver.tsne()


def update_args(args):
    with open(os.path.join("config", args.dataset + ".json")) as data_file:
        config = json.load(data_file)

    args.patch_size = config["patch_size"]

    args.epochs = config["epochs"]
    args.batch_size = config["batch_size"]
    args.lr = config["lr"]
    args.warmup = config["warmup"]
    args.image_size = config["image_size"]

    args.weight_decay = config["weight_decay"]
    args.hflip = config["hflip"]
    args.randomresizecrop = config["randomresizecrop"]
    args.padding = config["padding"]
    args.resizecrop = config["resizecrop"]
    args.cutout = config["cutout"]
    args.n_channels = config["n_channels"]
    args.n_classes = config["n_classes"]
    args.cm = config["cm"]
    args.mean = config["mean"]
    args.std = config["std"]

    args.data_path = os.path.join(args.data_path, args.dataset)
    args.model_path = os.path.join(args.model_path, args.dataset)
    args.tsne_path = os.path.join(args.tsne_path, args.dataset)

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PatchSwap')

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--method', type=str, default='patchswap', choices=['vanilla', 'patchswap'])

    parser.add_argument('--model_path', type=str, default='./saved_models/')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--tsne_path', type=str, default='./tsne/')

    parser.add_argument('--n_workers', type=int, default=4)

    args = parser.parse_args()
    args = update_args(args)

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))
    for k in dict(sorted(vars(args).items())).items(): print(k)
    print()
    main(args)
    end_time = datetime.datetime.now()
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    duration = end_time - start_time
    print("Duration: " + str(duration))
