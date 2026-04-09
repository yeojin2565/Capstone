"""
run_baseline.py

Random selection baseline 실험 실행
결과를 results_random.pkl로 저장

사용법:
    python run_baseline.py
"""
import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from random_strategy import FedAvgWithRandom
from DQN import K_SELECT
from he_utils import create_he_context, get_public_context


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    print("── Random Baseline 실험 ──")
    print(OmegaConf.to_yaml(cfg))
    
    # ── HE 컨텍스트 생성 (test.py와 동일 구조) ──────────
    print("── HE 컨텍스트 생성 중... ──")
    he_context_full         = create_he_context()
    he_context_public       = get_public_context(he_context_full)
    he_context_public_bytes = he_context_public.serialize()
    print("── HE 컨텍스트 생성 완료 ──")
 
    trainloaders, validationloaders, testloaders = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )
    

    client_fn = generate_client_fn(
        trainloaders, validationloaders, cfg.num_classes,
        he_context_bytes=he_context_public_bytes,
    )

    strategy = FedAvgWithRandom(
        k_select=cfg.num_clients_per_round_fit,
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloaders, he_context_full),
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        client_resources={"num_cpus": 0.5, "num_gpus": 0},
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 2, "num_gpus": 0, "include_dashboard": False},
    )
    
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results_random.pkl"

    results = {
        "history":      history,
        "dqn_metrics":  strategy.history_metrics,
        "method":       "random",
    }

    with open(str(Path(results_path)), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n결과 저장 완료: results_random.pkl")


if __name__ == "__main__":
    main()