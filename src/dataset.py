"""
src/dataset.py

CIFAR-10 Non-IID 데이터 분배 (Dirichlet α=0.5)
DataLoader 대신 Subset 반환 → Ray 직렬화 문제 해결
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop


def get_cifar10(data_path: str = "./data"):
    train_tf = Compose([
        RandomCrop(32, padding=4), RandomHorizontalFlip(),
        ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = Compose([
        ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return (CIFAR10(data_path, train=True,  download=True, transform=train_tf),
            CIFAR10(data_path, train=False, download=True, transform=test_tf))


def dirichlet_split(targets, num_clients: int, alpha: float = 0.5, seed: int = 42):
    np.random.seed(seed)
    num_classes    = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs        = np.where(np.array(targets) == c)[0]
        np.random.shuffle(idxs)
        props       = np.random.dirichlet([alpha] * num_clients)
        splits      = (props * len(idxs)).astype(int)
        splits[-1]  = len(idxs) - splits[:-1].sum()
        idx = 0
        for cid, n in enumerate(splits):
            client_indices[cid].extend(idxs[idx:idx + n].tolist())
            idx += n
    return client_indices


def prepare_dataset(num_clients=36, batch_size=32, val_ratio=0.1,
                    alpha=0.5, data_path="./data"):
    trainset, testset  = get_cifar10(data_path)
    targets            = [trainset[i][1] for i in range(len(trainset))]
    client_indices     = dirichlet_split(targets, num_clients, alpha)

    train_subsets, val_subsets = [], []
    for idxs in client_indices:
        if not idxs:
            idxs = [0]
        subset    = Subset(trainset, idxs)
        n_val     = max(1, int(val_ratio * len(subset)))
        for_train, for_val = random_split(
            subset, [len(subset) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_subsets.append(for_train)
        val_subsets.append(for_val)

    return train_subsets, val_subsets, DataLoader(testset, batch_size=128, shuffle=False)