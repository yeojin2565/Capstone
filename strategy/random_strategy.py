"""
random_strategy.py

FedAvg를 상속받아 configure_fit()을 오버라이드.
클라이언트를 완전 랜덤으로 K개 선택한다. (DQN 비교용 baseline)
"""

import time
import random
import numpy as np

from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes, Parameters

from DQN import K_SELECT

N_CLIENTS = 10


class FedAvgWithRandom(FedAvg):
    """
    클라이언트를 랜덤으로 K개 선택하는 baseline 전략.
    DQN 전략과 동일한 history_metrics 구조를 유지해서
    비교 그래프를 쉽게 그릴 수 있도록 한다.
    """

    def __init__(self, k_select: int = K_SELECT, **kwargs):
        super().__init__(**kwargs)
        self.k_select = k_select
        self._round_start = None
        self.history_metrics: list[dict] = []

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:

        self._round_start = time.time()

        all_clients = list(
            client_manager.sample(
                num_clients=client_manager.num_available(),
                min_num_clients=client_manager.num_available(),
            )
        )

        # 완전 랜덤 선택
        k = min(self.k_select, len(all_clients))
        selected_clients = random.sample(all_clients, k)
        selected_idx = [all_clients.index(c) for c in selected_clients]

        print(
            f"\n[Random] Round {server_round} | "
            f"선택된 클라이언트 인덱스: {selected_idx}"
        )

        config = {}
        if self.on_fit_config_fn:
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in selected_clients]

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

        metrics_list = [fit_res.metrics or {} for _, fit_res in results]

        accuracies = [m.get("accuracy", 0.0) for m in metrics_list]
        curr_accuracy = float(np.mean(accuracies)) if accuracies else 0.0

        he_latencies_norm = [
            float(np.clip(m.get("he_latency", 0.5) / 1.0, 0.0, 1.0))
            for m in metrics_list if m
        ]
        avg_he_latency_norm = float(np.mean(he_latencies_norm)) if he_latencies_norm else 0.5

        # DQN과 동일한 reward 공식 (공정한 비교)
        dropout_count = len(failures)
        reward = -avg_he_latency_norm - 0.3 * dropout_count

        self.history_metrics.append({
            "round":               server_round,
            "accuracy":            curr_accuracy,
            "avg_he_latency_norm": avg_he_latency_norm,
            "reward":              reward,
            "dropout_count":       dropout_count,
        })

        print(
            f"[Random] Round {server_round} 집계 완료 | "
            f"accuracy={curr_accuracy:.4f} | "
            f"avg_he_latency(norm)={avg_he_latency_norm:.4f} | "
            f"reward={reward:.4f}"
        )

        return aggregated_params, aggregated_metrics