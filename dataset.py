"""
dataset.py
 
CIFAR-10 Non-IID 데이터 분배
- Dirichlet(alpha=0.5) 분포로 클라이언트별 클래스 불균형 생성
- DataLoader 대신 Subset 반환 (Ray 직렬화 문제 해결)
- testloader는 서버 평가용으로 DataLoader 반환
"""


import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
)


def get_cifar10(data_path: str = "./data"):
    train_transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = CIFAR10(data_path, train=True,  download=True, transform=train_transform)
    testset  = CIFAR10(data_path, train=False, download=True, transform=test_transform)
    return trainset, testset


def dirichlet_split(targets, num_clients: int, alpha: float = 0.5, seed: int = 42): # defalut=42
    """
    Dirichlet 분포로 클라이언트별 인덱스 분배
    alpha 작을수록 더 non-IID (0.5 = 표준 설정)
    """
    np.random.seed(seed)
    num_classes    = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]
 
    for c in range(num_classes):
        class_idxs = np.where(np.array(targets) == c)[0]
        np.random.shuffle(class_idxs)
 
        proportions    = np.random.dirichlet([alpha] * num_clients)
        splits         = (proportions * len(class_idxs)).astype(int)
        splits[-1]     = len(class_idxs) - splits[:-1].sum()
 
        idx = 0
        for cid, n in enumerate(splits):
            client_indices[cid].extend(class_idxs[idx:idx + n].tolist())
            idx += n
 
    return client_indices


def prepare_dataset(
    # [FIXME]: 하드코딩
    num_clients: int = 100,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    alpha: float = 0.5,  
    data_path: str = "./data",
):
    """
    반환:
        train_subsets : list of Subset (클라이언트별 학습용)
        val_subsets   : list of Subset (클라이언트별 검증용)
        testloader    : DataLoader    (서버 전역 평가용)
    """
    trainset, testset = get_cifar10(data_path)
    targets = [trainset[i][1] for i in range(len(trainset))]
 
    client_indices = dirichlet_split(targets, num_clients, alpha)
 
    train_subsets, val_subsets = [], []
 
    for idxs in client_indices:
        if len(idxs) == 0:
            idxs = [0]
 
        subset    = Subset(trainset, idxs)
        num_total = len(subset)
        num_val   = max(1, int(val_ratio * num_total))
        num_train = num_total - num_val
 
        for_train, for_val = random_split(
            subset, [num_train, num_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_subsets.append(for_train)
        val_subsets.append(for_val)
 
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    return train_subsets, val_subsets, testloader
 

# ----- 클래스 분포 계산 -----
def compute_client_class_dist(train_subsets, num_classes: int = 10) -> np.ndarray:
    """
    클라이언트별 클래스 분포 계산
    반환: [N_CLIENTS, num_classes] (각 행은 정규화된 클래스 비율)
    """
    dist = np.zeros((len(train_subsets), num_classes), dtype=np.float32)

    for cid, subset in enumerate(train_subsets):
        # subset은 중첩 Subset이므로 실제 인덱스 추출
        parent = subset.dataset          # Subset(trainset, idxs)
        flat_indices = [parent.indices[i] for i in subset.indices]
        labels = [parent.dataset.targets[i] for i in flat_indices]

        for label in labels:
            dist[cid][label] += 1

        total = dist[cid].sum()
        if total > 0:
            dist[cid] /= total           # 정규화

    return dist


if __name__ == "__main__":
    train_subsets, val_subsets, testloader = prepare_dataset(num_clients=100)
    print(f"클라이언트 수: {len(train_subsets)}")
    print(f"테스트셋 크기: {len(testloader.dataset)}")
    for i in range(5):
        print(f"  Client {i:2d} | train={len(train_subsets[i])} | val={len(val_subsets[i])}")