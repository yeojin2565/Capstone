"""
dqn_strategy.py

[핵심 변경] Per-client scoring network에 맞게 state 관리 변경.

state 구조:
    _client_state: np.ndarray [N_CLIENTS, 2]
    state[i] = [he_latency_norm, data_size_norm] for client i

    초기값: [0.5, 0.5] (미관찰 → 중립값)
    관찰 즉시 실제값으로 업데이트

    → ScoringNetwork가 각 클라이언트의 [he, data]를 입력으로 받아
      "he 낮으면 높은 점수"를 빠르게 학습

리워드:
    - w3(HE) = 0.55 (강한 HE 패널티)
    - fast_bonus + slow_penalty (양방향 신호)
    - 커리큘럼 없음 (처음부터 일관된 HE 중심)
"""

import gc
import random
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import FitIns, FitRes, Parameters

from dqn import DQNAgent, K_SELECT, N_CLIENTS, EPSILON_DECAY, EPSILON_MIN
from he_simulator import HE_LATENCY_MAX

# hyperparameters
HE_MAX   = HE_LATENCY_MAX
DATA_MAX = 4000.0     # [REVIEW]: 왜 4000?

HE_BONUS_THRESHOLD = 0.10   # excellent + fast 그룹 (latency < 0.6s)
HE_BONUS_VALUE     = 0.25   # fast bonus 식에 사용
HE_SLOW_THRESHOLD  = 0.30   # slow + extreme 그룹 (latency > 1.8s)
HE_SLOW_PENALTY    = 0.20   # slow penalty 식에 사용


def default_client_state(n_clients: int = N_CLIENTS) -> np.ndarray:
    """
    [n_clients, 2]: [he_latency_norm, data_size_norm]
    초기값 0.5 (중립). 클라이언트가 관찰되면 실제값으로 갱신.
    """
    return np.full((n_clients, 2), 0.5, dtype=np.float32)


def compute_reward(
    metrics_list:  list[dict],
    dropout_count: int,
    prev_acc:      float,
    k_select:      int = K_SELECT,
    w1: float = 0.30,
    w2: float = 0.10,
    w3: float = 0.55,
    w4: float = 0.05,
) -> tuple[float, float]:
    # w1, w2, w3, w4: handling parameters
    # norms = normalized
    he_norms   = [np.clip(m.get("he_latency", 0.5) / HE_MAX,   0.0, 1.0) for m in metrics_list if m]
    data_norms = [np.clip(m.get("data_size",  500)  / DATA_MAX, 0.0, 1.0) for m in metrics_list if m]
    accs       = [m.get("accuracy", 0.0) for m in metrics_list if m]

    avg_he_norm = float(np.mean(he_norms)) if he_norms else 0.5
    curr_acc    = float(np.mean(accs))     if accs     else 0.0

    acc_gain      = curr_acc - prev_acc
    acc_gain_norm = float(np.tanh(acc_gain / 0.05))

    quality_bonuses   = [d * (1.0 - h) for d, h in zip(data_norms, he_norms)]
    avg_quality_bonus = float(np.mean(quality_bonuses)) if quality_bonuses else 0.0

    dropout_rate = dropout_count / max(k_select, 1)

    fast_count   = sum(1 for h in he_norms if h < HE_BONUS_THRESHOLD)
    fast_bonus   = HE_BONUS_VALUE * (fast_count / max(k_select, 1))

    slow_count   = sum(1 for h in he_norms if h > HE_SLOW_THRESHOLD)
    slow_penalty = HE_SLOW_PENALTY * (slow_count / max(k_select, 1))

    reward = (
          w1 * acc_gain_norm       # Δaccuracy
        + w2 * avg_quality_bonus   # data size 크고 HE latency 낮을 수록 bonus 상승
        - w3 * avg_he_norm         # HE latency
        - w4 * dropout_rate        
        + fast_bonus               
        - slow_penalty             
    )
    return reward, curr_acc


class FedAvgWithDQN(FedAvg):

    def __init__(self, dqn_agent: DQNAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent    = dqn_agent
        self.k_select = dqn_agent.k_select

        self._client_state = default_client_state(N_CLIENTS)   # [100, 2]
        self._prev_state   = self._client_state.flatten()       # [200]
        self._prev_action  = list(range(self.k_select))
        self._prev_acc     = 0.0

        self.history_metrics: list[dict] = []

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = list(client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        ))

        all_clients.sort(key=lambda c: int(c.cid))
        n = len(all_clients)

        raw_idx = self.agent.get_action(self._prev_state)
        seen = {}
        for i in raw_idx:
            seen[i % n] = None
        selected_idx = list(seen.keys())

        if len(selected_idx) < self.k_select:
            remaining = [i for i in range(n) if i not in seen]
            need      = min(self.k_select - len(selected_idx), len(remaining))
            selected_idx += random.sample(remaining, need)

        selected_clients  = [all_clients[i] for i in selected_idx]
        self._prev_action = selected_idx

        print(
            f"\n[DQN] Round {server_round} | "
            f"selected={selected_idx} | epsilon={self.agent.epsilon:.3f}"
        )

        config  = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)
        return [(c, fit_ins) for c in selected_clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        gc.collect()

        if not results:
            return aggregated_params, aggregated_metrics

        metrics_list  = [fit_res.metrics or {} for _, fit_res in results]
        dropout_count = sum(m.get("dropped", 0) for m in metrics_list) + len(failures)

        # ── state 업데이트: 관찰된 클라이언트만 갱신 ────────────────
        for m in metrics_list:
            cid = m.get("cid")
            if cid is None:
                print(f"[경고] cid 없는 메트릭 수신: {m}")
                continue
            if not (0 <= cid < N_CLIENTS):
                print(f"[경고] cid 범위 초과 (cid={cid}), 무시합니다.")
                continue
            he_norm   = float(np.clip(m.get("he_latency", 0.5) / HE_MAX,   0.0, 1.0))
            data_norm = float(np.clip(m.get("data_size",  500)  / DATA_MAX, 0.0, 1.0))
            self._client_state[cid] = [he_norm, data_norm]

        next_state = self._client_state.flatten()   # [200]

        reward, curr_acc = compute_reward(
            metrics_list, dropout_count, self._prev_acc, self.k_select
        )

        self.agent.append_sample(
            self._prev_state, self._prev_action, reward, next_state, False
        )
        loss = self.agent.train_step()

        # ── 라운드 기반 epsilon 감소 ──────────────────────────────────
        self.agent.epsilon = max(EPSILON_MIN, self.agent.epsilon * EPSILON_DECAY)

        avg_he      = float(np.mean([m.get("he_latency", 0.5) for m in metrics_list]))
        avg_he_norm = float(np.mean([
            np.clip(m.get("he_latency", 0.5) / HE_MAX, 0.0, 1.0)
            for m in metrics_list
        ]))
        avg_data = float(np.mean([m.get("data_size", 500) for m in metrics_list]))

        he_norms_log = [np.clip(m.get("he_latency", 0.5) / HE_MAX, 0.0, 1.0) for m in metrics_list]
        fast_cnt = sum(1 for h in he_norms_log if h < HE_BONUS_THRESHOLD)
        slow_cnt = sum(1 for h in he_norms_log if h > HE_SLOW_THRESHOLD)

        self.history_metrics.append({
            "round":               server_round,
            "accuracy":            curr_acc,
            "acc_gain":            curr_acc - self._prev_acc,
            "avg_he_latency":      avg_he,
            "avg_he_latency_norm": avg_he_norm,
            "avg_data_size":       avg_data,
            "reward":              reward,
            "dropout_count":       dropout_count,
            "dqn_loss":            loss,
            "epsilon":             self.agent.epsilon,
        })

        print(
            f"[DQN] Round {server_round} 완료 | "
            f"acc={curr_acc:.4f} (+{curr_acc - self._prev_acc:.4f}) | "
            f"he={avg_he:.3f}s | fast={fast_cnt} slow={slow_cnt} | "
            f"reward={reward:.4f} | eps={self.agent.epsilon:.3f} | "
            f"loss={f'{loss:.4f}' if loss else 'buffer 부족'}"
        )

        self._prev_acc   = curr_acc
        self._prev_state = next_state

        return aggregated_params, aggregated_metrics