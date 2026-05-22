"""
strategy/random_strategy.py

FedAvg 상속 → Random 기반 클라이언트 선택 (baseline)
"""

import random
import numpy as np

from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, Parameters


class FedAvgWithRandom(FedAvg):

    def __init__(self, k_select: int, **kwargs):
        super().__init__(**kwargs)
        self.k_select = k_select
        self.history_metrics: list[dict] = []

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:

        all_clients = list(client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        ))

        selected_clients = random.sample(all_clients, min(self.k_select, len(all_clients)))
        print(f"\n[Random] Round {server_round} | selected {len(selected_clients)} clients")

        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)
        return [(c, fit_ins) for c in selected_clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if results:
            metrics_list = [fit_res.metrics or {} for _, fit_res in results]
            curr_acc     = float(np.mean([m.get("accuracy",   0.0) for m in metrics_list]))
            avg_he       = float(np.mean([m.get("he_latency", 0.5) for m in metrics_list]))
            dropout_count = sum(m.get("dropped", 0) for m in metrics_list) + len(failures)

            self.history_metrics.append({
                "round":          server_round,
                "accuracy":       curr_acc,
                "avg_he_latency": avg_he,
                "dropout_count":  dropout_count,
            })

            print(f"[Random] Round {server_round} 완료 | acc={curr_acc:.4f} | he={avg_he:.3f}s")

        return aggregated_params, aggregated_metrics
