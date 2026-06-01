"""
rule_based_strategy.py - Data Size 우선 선택 Rule-based Baseline

리워드 가중치 dqn_strategy.py와 동일하게 통일 (공정한 비교):
    w1=0.40, w2=0.15, w3=0.40, w4=0.05
    HE_BONUS_VALUE=0.15, HE_SLOW_PENALTY=0.10
    tanh 스케일=0.05
"""

import gc
import numpy as np

from flwr.server.strategy import FedAvg
from flwr.common import FitIns, FitRes

from dqn import K_SELECT

HE_MAX             = 6.0
DATA_MAX           = 4000.0
HE_BONUS_THRESHOLD = 0.10
HE_BONUS_VALUE     = 0.15
HE_SLOW_THRESHOLD  = 0.30
HE_SLOW_PENALTY    = 0.10

DEFAULT_DATA_SIZE  = 500.0


class FedAvgWithRuleBased(FedAvg):

    def __init__(self, k_select: int = K_SELECT, **kwargs):
        super().__init__(**kwargs)
        self.k_select  = k_select
        self._prev_acc = 0.0
        self._known_data_size: dict[int, float] = {}
        self.history_metrics: list[dict] = []

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = list(client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        ))

        all_clients.sort(key=lambda c: int(c.cid))

        def get_data_size(c):
            return self._known_data_size.get(int(c.cid), DEFAULT_DATA_SIZE)

        sorted_clients   = sorted(all_clients, key=get_data_size, reverse=True)
        selected_clients = sorted_clients[:self.k_select]
        selected_idx     = [all_clients.index(c) for c in selected_clients]

        print(
            f"\n[RuleBased] Round {server_round} | "
            f"selected={selected_idx} | "
            f"data_sizes={[round(get_data_size(c)) for c in selected_clients]}"
        )

        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        return [(c, FitIns(parameters, config)) for c in selected_clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        gc.collect()

        if not results:
            return aggregated_params, aggregated_metrics

        metrics_list  = [fit_res.metrics or {} for _, fit_res in results]
        dropout_count = sum(m.get("dropped", 0) for m in metrics_list) + len(failures)

        for m in metrics_list:
            cid       = m.get("cid")
            data_size = m.get("data_size", DEFAULT_DATA_SIZE)
            if cid is not None and data_size > 0:
                self._known_data_size[int(cid)] = float(data_size)

        he_norms   = [np.clip(m.get("he_latency", 0.5) / HE_MAX,   0.0, 1.0) for m in metrics_list]
        data_norms = [np.clip(m.get("data_size",  500)  / DATA_MAX, 0.0, 1.0) for m in metrics_list]
        accs       = [m.get("accuracy", 0.0) for m in metrics_list]

        curr_acc    = float(np.mean(accs))
        avg_he      = float(np.mean([m.get("he_latency", 0.5) for m in metrics_list]))
        avg_he_norm = float(np.mean(he_norms))
        avg_data    = float(np.mean([m.get("data_size",  500)  for m in metrics_list]))

        acc_gain      = curr_acc - self._prev_acc
        acc_gain_norm = float(np.tanh(acc_gain / 0.05))

        quality_bonuses   = [d * (1.0 - h) for d, h in zip(data_norms, he_norms)]
        avg_quality_bonus = float(np.mean(quality_bonuses))

        dropout_rate = dropout_count / max(self.k_select, 1)

        fast_count   = sum(1 for h in he_norms if h < HE_BONUS_THRESHOLD)
        fast_bonus   = HE_BONUS_VALUE * (fast_count / max(self.k_select, 1))

        slow_count   = sum(1 for h in he_norms if h > HE_SLOW_THRESHOLD)
        slow_penalty = HE_SLOW_PENALTY * (slow_count / max(self.k_select, 1))

        reward = (
              0.40 * acc_gain_norm
            + 0.15 * avg_quality_bonus
            - 0.40 * avg_he_norm
            - 0.05 * dropout_rate
            + fast_bonus
            - slow_penalty
        )

        self.history_metrics.append({
            "round":               server_round,
            "accuracy":            curr_acc,
            "acc_gain":            acc_gain,
            "avg_he_latency":      avg_he,
            "avg_he_latency_norm": avg_he_norm,
            "avg_data_size":       avg_data,
            "reward":              reward,
            "dropout_count":       dropout_count,
        })

        print(
            f"[RuleBased] Round {server_round} 완료 | "
            f"acc={curr_acc:.4f} (+{acc_gain:.4f}) | "
            f"he={avg_he:.3f}s | data={avg_data:.0f} | reward={reward:.4f}"
        )

        self._prev_acc = curr_acc
        return aggregated_params, aggregated_metrics