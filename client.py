"""
client.py

Flower 클라이언트
"""

import time
from collections import OrderedDict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
import flwr as fl
from flwr.common import NDArray, Scalar

from model import Net, train, test
from he_simulator import init_base_latency, simulate_he_latency, simulate_dropout
import gc


_TRAIN_TRANSFORM = Compose([
    # data augmentation
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
_VAL_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def _unwrap_indices(subset) -> List[int]:
    """
    dataset.py의 random_split은 중첩 Subset을 만든다:
        for_train.dataset  = Subset(trainset, idxs)   ← 중간 Subset
        for_train.indices  = 중간 Subset 안 인덱스     ← trainset 직접 인덱스 X

    trainset 기준 실제 인덱스로 변환.
    """
    parent = subset.dataset   # Subset(trainset, idxs)
    return [parent.indices[i] for i in subset.indices]


def _make_loaders(train_indices: List[int], val_indices: List[int],
                  batch_size: int, data_path: str = "./data"):
    """trainset 기준 flat 인덱스로 DataLoader 생성 (쓰고 나면 del로 해제)"""
    trainset = CIFAR10(data_path, train=True, download=False, transform=_TRAIN_TRANSFORM)
    valset   = CIFAR10(data_path, train=True, download=False, transform=_VAL_TRANSFORM)

    trainloader = DataLoader(
        Subset(trainset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),  
    )
    valloader = DataLoader(
        Subset(valset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )
    return trainloader, valloader


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_indices: List[int],
        val_indices:   List[int],
        num_classes:   int,
        batch_size:    int,
        data_path:     str = "./data",
    ) -> None:
        super().__init__()

        self.cid           = cid
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model         = Net(num_classes).to(self.device)
        self.num_classes   = num_classes
        self.batch_size    = batch_size
        self.data_path     = data_path

        self.train_indices = train_indices
        self.val_indices   = val_indices

        self.base_he_latency = init_base_latency(cid, seed=42) # default=42
        print(f"Client {cid} using: {self.device}")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict(
            {k: torch.Tensor(v).to(self.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr       = config["lr"]
        momentum = config["momentum"]
        epochs   = config["local_epochs"]

        # ── [BUG-5 FIX] dropout 판정을 학습 전에 수행 ───────────────
        # 수정 전: 학습 완료 후 dropped를 판정 → 학습 결과가 집계에 포함됨
        # 수정 후: dropped=True면 학습 자체를 건너뛰고 서버 파라미터 그대로 반환.
        #          data_size=0 → FedAvg 가중 평균에서 가중치 0 → 사실상 집계 제외.
        dropped = simulate_dropout(self.cid)
        he_latency = simulate_he_latency(self.base_he_latency)

        if dropped:
            metrics = {
                "loss":          1.0,
                "accuracy":      0.0,
                "train_latency": 0.0,
                "he_latency":    float(he_latency),
                "data_size":     0,          # 가중치 0 → FedAvg 집계 제외
                "dropped":       1,
                "cid":           self.cid,
            }
            return self.get_parameters(config), 0, metrics

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        trainloader, valloader = _make_loaders(
            self.train_indices, self.val_indices,
            self.batch_size, self.data_path,
        )

        start_time    = time.time()
        train(self.model, trainloader, optimizer, epochs, self.device)
        train_latency = time.time() - start_time

        loss, accuracy = test(self.model, valloader, self.device)

        metrics = {
            "loss":          float(loss),
            "accuracy":      float(accuracy),
            "train_latency": float(train_latency),
            "he_latency":    float(he_latency),
            "data_size":     len(self.train_indices),
            "dropped":       0,
            "cid":           self.cid,
        }

        params_out = self.get_parameters(config)

        self.model.zero_grad()
        del optimizer, trainloader, valloader
        gc.collect()

        return params_out, len(self.train_indices), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        _, valloader = _make_loaders(
            self.train_indices, self.val_indices,
            self.batch_size, self.data_path,
        )
        loss, accuracy = test(self.model, valloader, self.device)

        del valloader
        gc.collect()

        return float(loss), len(self.val_indices), {"accuracy": accuracy}


def generate_client_fn(
    train_subsets,
    val_subsets,
    num_classes: int,
    batch_size:  int,
    data_path:   str = "./data",
):
    train_idx_list = [_unwrap_indices(s) for s in train_subsets]
    val_idx_list   = [_unwrap_indices(s) for s in val_subsets]

    def client_fn(cid: str):
        cid_int = int(cid)
        return FlowerClient(
            cid=cid_int,
            train_indices=train_idx_list[cid_int],
            val_indices=val_idx_list[cid_int],
            num_classes=num_classes,
            batch_size=batch_size,
            data_path=data_path,
        )
    return client_fn