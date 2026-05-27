"""
random_strategy.py - Random selection baseline
DQN과 동일한 reward 공식 사용 (공정한 비교)

수정 사항:
    [TUNE-1] reward 가중치·tanh 스케일·fast_bonus를 DQN과 동일하게 맞춤
"""

import random
import gc
import numpy as np

from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes, Parameters

from dqn import K_SELECT
from dqn_strategy import HE_MAX, DATA_MAX, HE_BONUS_THRESHOLD, HE_BONUS_VALUE, HE_SLOW_THRESHOLD, HE_SLOW_PENALTY

# [CHANGE]: 하드 코딩 고침
HE_MAX             = HE_MAX
DATA_MAX           = DATA_MAX
HE_BONUS_THRESHOLD = HE_BONUS_THRESHOLD
HE_BONUS_VALUE     = HE_BONUS_VALUE


class FedAvgWithRandom(FedAvg):

    def __init__(self, k_select: int = K_SELECT, **kwargs):
        super().__init__(**kwargs)
        self.k_select        = k_select
        self._prev_acc       = 0.0
        self.history_metrics: list[dict] = []

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients      = list(client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        ))
        k                = min(self.k_select, len(all_clients))
        selected_clients = random.sample(all_clients, k)
        selected_idx     = [all_clients.index(c) for c in selected_clients]

        print(f"\n[Random] Round {server_round} | selected={selected_idx}")

        config  = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
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

        he_norms   = [np.clip(m.get("he_latency", 0.5) / HE_MAX,   0.0, 1.0) for m in metrics_list]
        data_norms = [np.clip(m.get("data_size",  500)  / DATA_MAX, 0.0, 1.0) for m in metrics_list]
        accs       = [m.get("accuracy", 0.0) for m in metrics_list]

        curr_acc    = float(np.mean(accs))
        avg_he      = float(np.mean([m.get("he_latency", 0.5) for m in metrics_list]))
        avg_he_norm = float(np.mean(he_norms))
        avg_data    = float(np.mean([m.get("data_size", 500) for m in metrics_list]))

        acc_gain      = curr_acc - self._prev_acc
        acc_gain_norm = float(np.tanh(acc_gain / 0.03))   # DQN과 동일

        quality_bonuses   = [d * (1.0 - h) for d, h in zip(data_norms, he_norms)]
        avg_quality_bonus = float(np.mean(quality_bonuses))

        dropout_rate = dropout_count / max(self.k_select, 1)

        fast_count = sum(1 for h in he_norms if h < HE_BONUS_THRESHOLD)
        fast_bonus = HE_BONUS_VALUE * (fast_count / max(self.k_select, 1))
        
        slow_count   = sum(1 for h in he_norms if h > HE_SLOW_THRESHOLD)
        slow_penalty = HE_SLOW_PENALTY * (slow_count / max(self.k_select, 1))

        # [BUG/HACK]: dqn strategy와 다른 리워드 식 / 하드 코딩으로 동일하게 수정
        w1 = 0.30  # accuracy
        w2 = 0.10  # quality
        w3 = 0.55   # HE latency(defalut=0.55)
        w4 = 0.05  # dropout
        reward = (
              w1 * acc_gain_norm
            + w2 * avg_quality_bonus
            - w3 * avg_he_norm
            - w4 * dropout_rate
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
            f"[Random] Round {server_round} 완료 | "
            f"acc={curr_acc:.4f} (+{acc_gain:.4f}) | "
            f"he={avg_he:.3f}s | data={avg_data:.0f} | reward={reward:.4f}"
        )

        self._prev_acc = curr_acc
        return aggregated_params, aggregated_metrics