import random

from collections import OrderedDict
from typing import Dict
from flwr.common import NDArray, Scalar

import time
import torch
import flwr as fl

from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 num_class) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.device = torch.device("cpu")
        self.model = Net(num_class).to(self.device)


    
    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v).to(self.device) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        
        self.model.to(self.device)
        
        # 1. copy parameters sent by the sever into client's local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        
        # 2. loacal training 
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum = momentum)

        # do local training
        start_time = time.time()
        train(self.model, self.trainloader, optim, epochs, self.device)
        train_latency = time.time() - start_time

        # 3. val data로 loss / accuracy 측정
        loss, accuracy = test(self.model, self.valloader, self.device)
        
        # 4. he_latency
        # note: 지금은 더미 값이지만 추후 실제 측정 값으로 수정
        he_latency = random.uniform(0.01, 0.1) 

        # client feature(state feature) 
        metrics = { # 1 matrics per client
            "loss": float(loss),
            "accuracy": float(accuracy),
            "train_latency": float(train_latency),
            "he_latency": float(he_latency),
            "data_size": len(self.trainloader.dataset)
        }
        return self.get_parameters(config), len(self.trainloader.dataset), metrics


    def evaluate(self, parameters: NDArray, config: Dict[str, Scalar]):
        
        self.model.to(self.device)
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accuracy': accuracy}
    


def generate_client_fn(trainloeaders, valloaders, num_classes):

    def client_fn(cid: str):

        return FlowerClient(trainloader=trainloeaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_class=num_classes,
                            )

    return client_fn