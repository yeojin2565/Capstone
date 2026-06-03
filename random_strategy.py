"""
random_strategy.py - Random selection baseline
"""

import random
import gc
import numpy as np

from flwr.server.strategy import FedAvg
from flwr.common import FitIns

from dqn import K_SELECT
from dqn_strategy import (
    HE_MAX,
    DATA_MAX,
    HE_BONUS_THRESHOLD,
    HE_BONUS_VALUE,
    HE_SLOW_THRESHOLD,
    HE_SLOW_PENALTY,
)


class FedAvgWithRandom(FedAvg):

    def __init__(
        self,
        client_class_dist,
        k_select: int = K_SELECT,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.k_select = k_select
        self._prev_acc = 0.0
        self.history_metrics: list[dict] = []

        # entropy용
        self.client_class_dist = client_class_dist
        self._selected_class_dist = np.zeros(10, dtype=np.float32)

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = list(client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        ))

        k = min(self.k_select, len(all_clients))
        selected_clients = random.sample(all_clients, k)

        # client index 추출
        selected_idx = [all_clients.index(c) for c in selected_clients]

        # ── entropy용 class distribution 저장 ───────────────
        selected_class_dist       = self.client_class_dist[selected_idx]
        self._selected_class_dist = selected_class_dist.mean(axis=0)

        print(
            f"\n[Random] Round {server_round} | "
            f"selected={selected_idx}"
        )

        config = (
            self.on_fit_config_fn(server_round)
            if self.on_fit_config_fn
            else {}
        )

        return [
            (c, FitIns(parameters, config))
            for c in selected_clients
        ]

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )
        gc.collect()

        if not results:
            return aggregated_params, aggregated_metrics

        metrics_list = [
            fit_res.metrics or {}
            for _, fit_res in results
        ]

        dropout_count = (
            sum(m.get("dropped", 0) for m in metrics_list)
            + len(failures)
        )

        he_norms = [
            np.clip(m.get("he_latency", 0.5) / HE_MAX, 0.0, 1.0)
            for m in metrics_list
        ]

        data_norms = [
            np.clip(m.get("data_size", 500) / DATA_MAX, 0.0, 1.0)
            for m in metrics_list
        ]

        accs = [
            m.get("accuracy", 0.0)
            for m in metrics_list
        ]

        curr_acc = float(np.mean(accs))

        avg_he = float(np.mean([
            m.get("he_latency", 0.5)
            for m in metrics_list
        ]))

        avg_he_norm = float(np.mean(he_norms))

        avg_data = float(np.mean([
            m.get("data_size", 500)
            for m in metrics_list
        ]))

        acc_gain = curr_acc - self._prev_acc
        acc_gain_norm = float(np.tanh(acc_gain / 0.05)) # DQN과 동일

        quality_bonuses = [
            d * (1.0 - h)
            for d, h in zip(data_norms, he_norms)
        ]
        avg_quality_bonus = float(np.mean(quality_bonuses))

        dropout_rate = dropout_count / max(self.k_select, 1)

        fast_count = sum(
            1 for h in he_norms
            if h < HE_BONUS_THRESHOLD
        )
        fast_bonus = (
            HE_BONUS_VALUE
            * (fast_count / max(self.k_select, 1))
        )

        slow_count = sum(
            1 for h in he_norms
            if h > HE_SLOW_THRESHOLD
        )
        slow_penalty = (
            HE_SLOW_PENALTY
            * (slow_count / max(self.k_select, 1))
        )

        # DQN과 동일 reward
        w1 = 0.30
        w2 = 0.10
        w3 = 0.55
        w4 = 0.05

        reward = (
              w1 * acc_gain_norm
            + w2 * avg_quality_bonus
            - w3 * avg_he_norm
            - w4 * dropout_rate
            + fast_bonus
            - slow_penalty
        )

        # ── history logging ───────────────────────────
        self.history_metrics.append({
            "round":               server_round,
            "accuracy":            curr_acc,
            "acc_gain":            acc_gain,
            "avg_he_latency":      avg_he,
            "avg_he_latency_norm": avg_he_norm,
            "avg_data_size":       avg_data,
            "reward":              reward,
            "dropout_count":       dropout_count,
            "class_distribution":  self._selected_class_dist.tolist(),
        })

        print(
            f"[Random] Round {server_round} 완료 | "
            f"acc={curr_acc:.4f} (+{acc_gain:.4f}) | "
            f"he={avg_he:.3f}s | "
            f"data={avg_data:.0f} | "
            f"reward={reward:.4f}"
        )

        self._prev_acc = curr_acc
        return aggregated_params, aggregated_metrics