"""
dqn_strategy.py

[변경 이력]
    [BUG-1]  configure_fit: CID 기준 정렬, 중복 제거 + k_select 미달 보충.
    [BUG-2]  aggregate_fit: cid 누락/범위 초과 메트릭 경고 로그.
    [TUNE-1] epsilon: aggregate_fit() 라운드 기반 감소.
    [TUNE-2] 리워드: HE 패널티(w3=0.55) + fast_bonus + slow_penalty.
    [FEAT]   client_state 5피처 복원:
             [loss_norm, acc_norm, train_latency_norm, he_latency_norm, data_size_norm]
             = 원본 코드의 5개 피처를 Per-client scoring에 맞게 그대로 복원.
             Per-client 구조이므로 피처가 섞이지 않고 클라이언트별 독립 학습.

             ScoringNetwork가 학습하는 것:
               he_norm  ↓ → score ↑  (HE 빠른 클라이언트 선호)
               data_norm ↑ → score ↑  (데이터 많은 클라이언트 선호)
               acc_norm  ↑ → score ↑  (정확도 높은 클라이언트 선호)
               loss_norm ↓ → score ↑  (손실 낮은 클라이언트 선호)
               train_lat ↓ → score ↑  (학습 빠른 클라이언트 선호)
"""

import gc
import random
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import FitIns, FitRes, Parameters

from dqn import DQNAgent, K_SELECT, N_CLIENTS, EPSILON_DECAY, EPSILON_MIN

# 원본 정규화 스케일 복원
SCALE = np.array([5.0, 1.0, 10.0, 6.0, 4000.0], dtype=np.float32)
# 피처 순서: [loss, accuracy, train_latency, he_latency, data_size]

HE_MAX   = 6.0
DATA_MAX = 4000.0

HE_BONUS_THRESHOLD = 0.10
HE_BONUS_VALUE     = 0.25
HE_SLOW_THRESHOLD  = 0.30
HE_SLOW_PENALTY    = 0.20

# 피처 인덱스 (가독성)
IDX_LOSS      = 0
IDX_ACC       = 1
IDX_TRAIN_LAT = 2
IDX_HE        = 3
IDX_DATA      = 4


def default_client_state(n_clients: int = N_CLIENTS) -> np.ndarray:
    """
    5피처 초기값: [loss=1.0, acc=0.5, train_lat=0.5, he=0.5, data=0.5]
    원본 코드와 동일한 초기값 사용.
    관찰될 때마다 실제 측정값으로 갱신됨.
    """
    return np.tile(
        [1.0, 0.5, 0.5, 0.5, 0.5], (n_clients, 1)
    ).astype(np.float32)


def normalize_row(m: dict) -> np.ndarray:
    """원본 코드와 동일한 정규화."""
    row = np.array([
        m.get("loss",          1.0),
        m.get("accuracy",      0.5),
        m.get("train_latency", 0.5),
        m.get("he_latency",    0.5),
        m.get("data_size",     1000.0),
    ], dtype=np.float32)
    return np.clip(row / SCALE, 0.0, 1.0)


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
          w1 * acc_gain_norm
        + w2 * avg_quality_bonus
        - w3 * avg_he_norm
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

        self._client_state = default_client_state(N_CLIENTS)   # [100, 5]
        self._prev_state   = self._client_state.flatten()       # [500]
        self._prev_action  = list(range(self.k_select))
        self._prev_acc     = 0.0

        self.history_metrics: list[dict] = []

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = list(client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        ))

        # [BUG-1 FIX] CID 기준 정렬
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

        # ── [BUG-2 FIX + FEAT] 5피처 state 업데이트 ─────────────────
        # 원본 normalize_row() 사용: 5개 피처 모두 정규화하여 저장
        for m in metrics_list:
            cid = m.get("cid")
            if cid is None:
                print(f"[경고] cid 없는 메트릭 수신: {m}")
                continue
            if not (0 <= cid < N_CLIENTS):
                print(f"[경고] cid 범위 초과 (cid={cid}), 무시합니다.")
                continue
            self._client_state[cid] = normalize_row(m)   # [loss, acc, train_lat, he, data]

        next_state = self._client_state.flatten()   # [500]

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