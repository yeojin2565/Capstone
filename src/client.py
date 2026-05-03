# clinent.py

import time
import pickle
from collections import OrderedDict
from typing import Dict

import numpy as np
import tenseal as ts
import torch
import flwr as fl
from flwr.common import NDArray, Scalar

from model import Net, train, test
from he_utils import create_he_context, get_public_context, encrypt_weights


# ── 클라이언트 성능 그룹 ───────────────────────────────
# cid 기반으로 HE 연산 속도 그룹 배정(시뮬레이션 내 고정 특성)
def get_he_group(cid: int) -> str:
    if cid < 10:
        return "fast"
    elif cid < 20:
        return "medium"
    else:
        return "slow"
    
# 그룹별 연산 지연 배수 (poly_modulus_degree를 크게 하거나 sleep으로 시뮬레이션)
HE_DELAY = {
    "fast": 0.0,        # 추가 지연 없음
    "medium": 0.05,     # 50ms 추가
    "slow": 0.15,       # 150ms 추가
}


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 cid: int,
                 trainloader,
                 valloader,
                 num_class,
                 he_context_byte: bytes) -> None:
        super().__init__()

        
        self.cid         = cid
        self.trainloader = trainloader
        self.valloader   = valloader
        self.he_group    = get_he_group(cid)
        self.he_context  = ts.context_from(he_context_byte)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        lr       = config['lr']
        momentum = config['momentum']
        epochs   = config['local_epochs']
        
        # 2. loacal training 
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum = momentum)

        # do local training
        start_time = time.time()
        train(self.model, self.trainloader, optim, epochs, self.device)
        train_latency = time.time() - start_time

        # 3. val data로 loss / accuracy 측정
        loss, accuracy = test(self.model, self.valloader, self.device)
        
        # 4. he_latency
        weights = self.get_parameters(config)
        shapes  = [w.shape for w in weights]
        
        he_start = time.time()
        
        # 그룹별 추가 지연 (느린 기기 시뮬레이션)
        time.sleep(HE_DELAY[self.he_group])
        
        encrypted = encrypt_weights(weights, self.he_context)
        
        he_latency = time.time() - he_start  # 실제 암호화 소요 시간 (초)
        
        # 암호화된 가중치를 bytes로 직렬화 (Flower matrics는 기본 타입만 허용)
        enc_bytes = pickle.dumps({
            "enc_weights": [enc.serialize() for enc in encrypted],
            "shapes": shapes,
        })

        # client feature(state feature) 
        metrics = { # 1 matrics per client
            "loss":          float(loss),
            "accuracy":      float(accuracy),
            "train_latency": float(train_latency),
            "he_latency":    float(he_latency),
            "data_size":     len(self.trainloader.dataset),
            "enc_weights":   enc_bytes,  # 암호화된 가중치 전달
            "cid":           self.cid,
        }
        
        # fit 반환: 평문 가중치도 함께 반환 (FedAvg fallback용)
        return self.get_parameters(config), len(self.trainloader.dataset), metrics


    def evaluate(self, parameters: NDArray, config: Dict[str, Scalar]):
        
        self.model.to(self.device)
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accuracy': accuracy}
    


def generate_client_fn(trainloeaders, valloaders, num_classes, he_context_bytes: bytes):

    def client_fn(cid: str):

        return FlowerClient(
            cid=int(cid),
            trainloader=trainloeaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_class=num_classes,
            he_context_byte=he_context_bytes,
        )

    return client_fn