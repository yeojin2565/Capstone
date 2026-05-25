"""
train_random.py

Random selection baseline 실험 실행
결과: outputs/{날짜}/{시간}/results_random.pkl

수정 사항:
    [BUG-8 경미] ray.init(): 이미 초기화된 경우 중복 호출 방지 (ray.is_initialized() 가드)
    [BUG-9 경미] 미사용 import torch 제거
    [TUNE-5 FIX] DQN과 공정한 비교를 위해 동일한 num_rounds 사용 권장.
                 conf/base.yaml num_rounds: 200
"""

import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import flwr as fl
import ray

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from dqn import K_SELECT
from random_strategy import FedAvgWithRandom

# ── [BUG-8 FIX] 중복 초기화 방지 ───────────────────────────────────
if not ray.is_initialized():
    ray.init(
        num_cpus=2,
        object_store_memory=256 * 1024**2,  # 256MB
        _system_config={
            "max_io_workers": 1,
            "automatic_object_spilling_enabled": False,
        }
    )


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print("── Random Baseline ──")
    print(OmegaConf.to_yaml(cfg))

    # 1. 데이터 준비
    train_subsets, val_subsets, testloader = prepare_dataset(
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
    )

    # 2. 클라이언트 정의
    client_fn = generate_client_fn(
        train_subsets, val_subsets,
        num_classes=cfg.num_classes,
        batch_size=cfg.batch_size,
        data_path="./data",
    )

    # 3. 전략
    strategy = FedAvgWithRandom(
        k_select=cfg.num_clients_per_round_fit,
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )

    # 4. 시뮬레이션
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        client_resources={"num_cpus": 2},
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    # 5. 저장
    save_path    = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results_random.pkl"

    with open(str(results_path), "wb") as f:
        pickle.dump(
            {"history": history, "dqn_metrics": strategy.history_metrics, "method": "random"},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"\n결과 저장 완료: {results_path}")


if __name__ == "__main__":
    main()