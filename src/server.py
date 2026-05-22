"""src/server.py"""

from collections import OrderedDict
from omegaconf import DictConfig
import torch
from src.model import Net, test  # noqa: E402 (절대 import 유지 — 서버는 Ray 워커 아님)


def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {"lr": config.lr, "momentum": config.momentum,
                "local_epochs": config.local_epochs}
    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Net(num_classes).to(device)

    def evaluate_fn(server_round, parameters, config):
        state_dict = OrderedDict(
            {k: torch.Tensor(v).to(device)
             for k, v in zip(model.state_dict().keys(), parameters)}
        )
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(model, testloader, device)
        print(f"[Server] Round {server_round} | loss={loss:.4f} | acc={accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    return evaluate_fn