"""
server.py

서버 설정
- fit config 전달
- 글로벌 모델 평가
"""

from collections import OrderedDict
from omegaconf import DictConfig

import torch
from model import Net, test


def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {
            "lr":           config.lr,
            "momentum":     config.momentum,
            "local_epochs": config.local_epochs,
        }
    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = Net(num_classes).to(device)

    def evaluate_fn(server_round: int, parameters, config):
        params_dict = zip(global_model.state_dict().keys(), parameters)
        state_dict  = OrderedDict(
            {k: torch.Tensor(v).to(device) for k, v in params_dict}
        )
        global_model.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(global_model, testloader, device)

        print(f"[Server] Round {server_round} | loss={loss:.4f} | accuracy={accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    return evaluate_fn