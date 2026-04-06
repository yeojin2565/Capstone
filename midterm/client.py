import random
from collections import OrderedDict
from typing import Dict

from flwr.common import NDArray, Scalar

import time
import torch
import flwr as fl

from model import Net, train, test
from crypto import encrypt_weights


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_class) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.device = torch.device("cpu")
        self.model = Net(num_class).to(self.device)

        # 마지막 state 저장
        self.last_state = None

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.Tensor(v).to(self.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):

        self.model.to(self.device)

        # 이전 global weight 저장 (delta 계산용)
        old_weights = parameters

        # global 적용
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # local training
        start_time = time.time()
        train(self.model, self.trainloader, optim, epochs, self.device)
        train_latency = time.time() - start_time

        # validation
        loss, accuracy = test(self.model, self.valloader, self.device)

        # 현재 weight
        new_weights = self.get_parameters(config)


        # HE 암호화
        start_he = time.time()
        encrypted_weights = encrypt_weights(new_weights)
        he_latency = time.time() - start_he

        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "train_latency": float(train_latency),
            "he_latency": float(he_latency),
            "data_size": len(self.trainloader.dataset)
        }

        self.last_state = metrics

        # delta를 반환 (weight 아님)
        return encrypted_weights, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: NDArray, config: Dict[str, Scalar]):

        self.model.to(self.device)
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accuracy': accuracy}

    def get_state(self):
        return self.last_state


def generate_client_fn(trainloaders, valloaders, num_classes):

    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_class=num_classes,
        )

    return client_fn