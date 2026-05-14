"""experiments/train_a2c.py — A2C 실험 실행"""

import pickle
import torch
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import flwr as fl

from src.dataset import prepare_dataset
from src.client import generate_client_fn
from src.server import get_on_fit_config, get_evaluate_fn
from rl.a2c import A2CAgent, N_CLIENTS, N_FEATURES, K_SELECT
from strategy.a2c_strategy import FedAvgWithA2C


@hydra.main(config_path="../conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_subsets, val_subsets, testloader = prepare_dataset(
        num_clients=cfg.num_clients, batch_size=cfg.batch_size,
    )
    client_fn = generate_client_fn(
        train_subsets, val_subsets, cfg.num_classes, cfg.batch_size,
    )

    agent    = A2CAgent(n_features=N_FEATURES, n_clients=N_CLIENTS, k_select=cfg.num_clients_per_round_fit)
    save_path = HydraConfig.get().runtime.output_dir

    strategy = FedAvgWithA2C(
        agent=agent,
        log_dir=save_path,
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )

    n_gpus = 1 if torch.cuda.is_available() else 0

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        client_resources={"num_cpus": 1, "num_gpus": 0.1 if n_gpus > 0 else 0},
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        ray_init_args={
            "num_cpus": 4, "num_gpus": n_gpus,
            "include_dashboard": False,
        },
    )

    results_path = Path(save_path) / "results.pkl"
    with open(str(results_path), "wb") as f:
        pickle.dump(
            {"history": history, "dqn_metrics": strategy.history_metrics, "method": "a2c"},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )

    agent.save(str(Path(save_path) / "a2c_model.pth"))
    print(f"\n결과 저장 완료: {results_path}")


if __name__ == "__main__":
    main()