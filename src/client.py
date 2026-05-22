"""
src/client.py

Flower 클라이언트
- HE 실제 연산 없음
- 가우시안 분포 기반 HE latency 시뮬레이션
- dropout 시뮬레이션
- recent_dropout_rate는 서버에서 config로 전달
"""

import sys
import os
# Ray 워커 프로세스에서도 프로젝트 루트를 찾을 수 있도록 설정
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
from collections import OrderedDict
from typing import Dict

import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import NDArray, Scalar

from src.model import Net, train, test
from src.he_simulator import init_base_latency, simulate_he_latency, simulate_dropout


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, train_subset, val_subset, num_classes, batch_size):
        super().__init__()
        self.cid         = cid
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = Net(num_classes).to(self.device)
        self.trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.valloader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)
        self.base_he     = init_base_latency(cid, seed=42)

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            {k: torch.Tensor(v).to(self.device)
             for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config["lr"], momentum=config["momentum"],
        )
        t0 = time.time()
        train(self.model, self.trainloader, optimizer, config["local_epochs"], self.device)
        train_latency = time.time() - t0

        loss, accuracy = test(self.model, self.valloader, self.device)
        he_latency     = simulate_he_latency(self.base_he)
        dropped        = simulate_dropout(self.cid)
        dropout_rate   = float(config.get("recent_dropout_rate", 0.0))

        metrics = {
            "loss":                float(loss),
            "accuracy":            float(accuracy),
            "train_latency":       float(train_latency),
            "he_latency":          float(he_latency),
            "data_size":           len(self.trainloader.dataset),
            "recent_dropout_rate": dropout_rate,
            "dropped":             int(dropped),
            "cid":                 self.cid,
        }
        return self.get_parameters(config), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: NDArray, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}


def generate_client_fn(train_subsets, val_subsets, num_classes, batch_size):
    def client_fn(cid: str):
        i = int(cid)
        return FlowerClient(i, train_subsets[i], val_subsets[i], num_classes, batch_size)
    return client_fn