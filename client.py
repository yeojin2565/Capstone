"""
client.py

Flower 클라이언트
- HE 실제 연산 없음
- 클라이언트 초기화 시 가우시안 분포로 base HE latency 결정
- 매 라운드 전송 노이즈 추가
- dropout 시뮬레이션
- recent_dropout_rate는 서버(dqn_strategy)에서 관리 후 config로 전달
"""

import time
from collections import OrderedDict
from typing import Dict

import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import NDArray, Scalar

from model import Net, train, test
from he_simulator import init_base_latency, simulate_he_latency, simulate_dropout


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_subset,
        val_subset,
        num_classes: int,
        batch_size: int,
    ) -> None:
        super().__init__()

        self.cid        = cid
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = Net(num_classes).to(self.device)
        self.batch_size = batch_size

        self.trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.valloader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)

        # 클라이언트 고유 HE latency (초기화 시 1회 결정)
        self.base_he_latency = init_base_latency(cid, seed=42)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict(
            {k: torch.Tensor(v).to(self.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr       = config["lr"]
        momentum = config["momentum"]
        epochs   = config["local_epochs"]

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum
        )

        start_time = time.time()
        train(self.model, self.trainloader, optimizer, epochs, self.device)
        train_latency = time.time() - start_time

        loss, accuracy = test(self.model, self.valloader, self.device)

        # HE latency 시뮬레이션: base(고정) + 전송 노이즈(매 라운드)
        he_latency = simulate_he_latency(self.base_he_latency)

        # dropout 시뮬레이션
        dropped = simulate_dropout(self.cid)

        # recent_dropout_rate: 서버가 config로 전달
        # 첫 라운드 등 없을 경우 0.0으로 fallback
        recent_dropout_rate = float(config.get("recent_dropout_rate", 0.0))

        metrics = {
            "loss":                float(loss),
            "accuracy":            float(accuracy),
            "train_latency":       float(train_latency),
            "he_latency":          float(he_latency),
            "data_size":           len(self.trainloader.dataset),
            "recent_dropout_rate": recent_dropout_rate,
            "dropped":             int(dropped),
            "cid":                 self.cid,
        }

        return self.get_parameters(config), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: NDArray, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}


def generate_client_fn(
    train_subsets,
    val_subsets,
    num_classes: int,
    batch_size: int,
):
    def client_fn(cid: str):
        cid_int = int(cid)
        return FlowerClient(
            cid=cid_int,
            train_subset=train_subsets[cid_int],
            val_subset=val_subsets[cid_int],
            num_classes=num_classes,
            batch_size=batch_size,
        )
    return client_fn