# server.py

import pickle
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from model import Net, test
from he_utils import he_fedavg, decrypt_weights


# ── fit config ────────────────────────────────────────
def get_on_fit_config(config: DictConfig):
    """매 라운드 클라이언트에게 lr, momentum, epochs 전달"""
    def fit_config_fn(server_round: int):
        return{'lr': config.lr, 
               'momentum': config.momentum,
               'local_epochs': config.local_epochs
               }

    return fit_config_fn


# ── HE FedAvg + 복호화 ────────────────────────────────
def aggregate_he_weights(
    results: List[Tuple],
    he_context,          # 비밀키 포함 컨텍스트 (서버만 보유)
) -> Optional[List[np.ndarray]]:
    """
    클라이언트 metrics에서 암호화된 가중치 추출
    → HE FedAvg (암호화 상태 덧셈)
    → 서버에서 복호화
    → 평균 가중치 반환
 
    HE 연산 실패 시 None 반환 → 평문 FedAvg로 fallback
    """
    enc_list = []
    shape_list = None
 
    for _, fit_res in results:
        enc_bytes = fit_res.metrics.get("enc_weights")
        if enc_bytes is None:
            return None
        try:
            data       = pickle.loads(enc_bytes)
            shapes      = data["shapes"]
 
            # 컨텍스트 링크 복원 (직렬화 시 컨텍스트 분리됨)
            import tenseal as ts
            enc_weights = [
                ts.ckks_vector_from(he_context, b)
                for b in data["enc_weights"]
            ]
 
            enc_list.append(enc_weights)
            if shape_list is None:
                shape_list = shapes
 
        except Exception as e:
            print(f"[HE] 복호화 준비 실패: {e}")
            return None
 
    if not enc_list:
        return None
 
    try:
        # 암호화 상태로 FedAvg
        averaged_enc = he_fedavg(enc_list, n_clients=len(enc_list))
 
        # 서버에서 복호화
        averaged_weights = decrypt_weights(averaged_enc, shape_list)
        print(f"[HE] FedAvg 복호화 완료 ({len(enc_list)}개 클라이언트)")
        return averaged_weights
 
    except Exception as e:
        print(f"[HE] FedAvg 실패: {e}, 평문 FedAvg로 fallback")
        return None
    

# ── 글로벌 모델 평가 ──────────────────────────────────
def get_evaluate_fn(num_classes: int, testloader, he_context):
    """
    매 라운드 종료 후 서버가 global 모델을 testset으로 직접 평가
    HE 복호화된 가중치로 모델을 업데이트한 뒤 평가 수행
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = Net(num_classes).to(device)
    
    def evaluate_fn(server_round: int, parameters, config):
        # 파라미터 -> 모델 업데이트
        params_dict = zip(global_model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.Tensor(v).to(device) for k, v in params_dict}
        )
        global_model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(global_model, testloader, device)

        return loss, {"accuracy": accuracy}
    
    
    return evaluate_fn