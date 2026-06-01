"""
client.py

Flower 클라이언트

수정 사항:
    [BUG-4]  _make_loaders: CPU 환경에서 pin_memory=False 고정
    [BUG-5]  fit(): dropout 판정을 학습 전에 수행, data_size=0으로 FedAvg 집계 제외
    [BUG-12] fit(): simulate_he_latency에 cid, round_num 전달 → 재현성 있는 독립 RNG
    [BUG-13] fit(): simulate_dropout에 round_num 전달 → 재현성 있는 독립 RNG
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
        for_train.dataset  = Subset(trainset, idxs)
        for_train.indices  = 중간 Subset 안 인덱스

    trainset 기준 실제 인덱스로 변환.
    """
    parent = subset.dataset
    return [parent.indices[i] for i in subset.indices]


def _make_loaders(train_indices: List[int], val_indices: List[int],
                  batch_size: int, data_path: str = "./data"):
    """trainset 기준 flat 인덱스로 DataLoader 생성"""
    trainset = CIFAR10(data_path, train=True, download=False, transform=_TRAIN_TRANSFORM)
    valset   = CIFAR10(data_path, train=True, download=False, transform=_VAL_TRANSFORM)

    trainloader = DataLoader(
        Subset(trainset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,   # CPU-only 환경
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
        self.device        = torch.device("cpu")
        self.model         = Net(num_classes).to(self.device)
        self.num_classes   = num_classes
        self.batch_size    = batch_size
        self.data_path     = data_path

        self.train_indices = train_indices
        self.val_indices   = val_indices

        self.base_he_latency = init_base_latency(cid, seed=42)
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

        lr        = config["lr"]
        momentum  = config["momentum"]
        epochs    = config["local_epochs"]
        round_num = int(config.get("server_round", 0))   # ← 라운드 번호 수신

        # [BUG-12/13 FIX] cid + round_num → 독립 RNG → 재현성 보장
        # Ray 병렬 환경에서 실행 순서와 무관하게 동일 결과
        dropped    = simulate_dropout(self.cid, round_num=round_num)
        he_latency = simulate_he_latency(
            self.base_he_latency, cid=self.cid, round_num=round_num
        )

        if dropped:
            metrics = {
                "loss":          1.0,
                "accuracy":      0.0,
                "train_latency": 0.0,
                "he_latency":    float(he_latency),
                "data_size":     0,
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
