"""
random_strategy.py

Random selection baseline
"""

import random
import time
import numpy as np

from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes, Parameters

from dqn import K_SELECT


class FedAvgWithRandom(FedAvg):

    def __init__(self, k_select: int = K_SELECT, **kwargs):
        super().__init__(**kwargs)
        self.k_select        = k_select
        self.history_metrics: list[dict] = []

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:

        all_clients      = list(client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        ))
        k                = min(self.k_select, len(all_clients))
        selected_clients = random.sample(all_clients, k)
        selected_idx     = [all_clients.index(c) for c in selected_clients]

        print(f"\n[Random] Round {server_round} | selected={selected_idx}")

        config  = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)
        return [(c, fit_ins) for c in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ):
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if not results:
            return aggregated_params, aggregated_metrics

        metrics_list  = [fit_res.metrics or {} for _, fit_res in results]
        dropout_count = sum(m.get("dropped", 0) for m in metrics_list) + len(failures)

        curr_acc = float(np.mean([m.get("accuracy", 0.0) for m in metrics_list]))
        avg_he   = float(np.mean([m.get("he_latency", 0.5) for m in metrics_list]))

        # DQN과 동일한 reward 공식으로 계산 (공정한 비교)
        alpha = 0.3
        beta = 0.3
        
        he_norms = [np.clip(m.get("he_latency", 0.5) / 1.5, 0.0, 1.0) for m in metrics_list]
        reward   = -float(np.mean(he_norms)) + alpha * curr_acc - beta * dropout_count

        self.history_metrics.append({
            "round":          server_round,
            "accuracy":       curr_acc,
            "avg_he_latency": avg_he,
            "reward":         reward,
            "dropout_count":  dropout_count,
        })

        print(
            f"[Random] Round {server_round} 완료 | "
            f"acc={curr_acc:.4f} | he={avg_he:.3f}s | reward={reward:.4f}"
        )

        return aggregated_params, aggregated_metrics