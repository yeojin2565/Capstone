"""
dqn_strategy.py

수정 사항:
    [BUG-1 치명] configure_fit: all_clients를 CID 기준 정렬 → 라운드마다 순서 달라지는 문제 제거
    [BUG-1 치명] selected_idx 중복 제거 + k_select 미달 시 보충
    [BUG-2 중간] aggregate_fit: cid 누락/범위 초과 메트릭에 경고 로그 추가
"""

import gc
import random
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes, Parameters

from dqn import DQNAgent, K_SELECT, N_CLIENTS, STATE_SIZE

SCALE    = np.array([5.0, 1.0, 10.0, 6.0, 4000.0], dtype=np.float32)
HE_MAX   = 6.0
DATA_MAX = 4000.0


def default_client_state(n_clients: int = N_CLIENTS) -> np.ndarray:
    return np.tile(
        [1.0, 0.5, 0.5, 0.5, 0.5], (n_clients, 1)
    ).astype(np.float32)


def normalize_row(m: dict) -> np.ndarray:
    row = np.array([
        m.get("loss",          1.0),
        m.get("accuracy",      0.5),
        m.get("train_latency", 0.5),
        m.get("he_latency",    0.5),
        m.get("data_size",     1000.0),
    ], dtype=np.float32)
    return np.clip(row / SCALE, 0.0, 1.0)


def compute_reward(
    metrics_list: list[dict],
    dropout_count: int,
    prev_acc: float,
    k_select: int = K_SELECT,
    w1: float = 0.4,
    w2: float = 0.3,
    w3: float = 0.2,
    w4: float = 0.1,
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

    reward = (
          w1 * acc_gain_norm
        + w2 * avg_quality_bonus
        - w3 * avg_he_norm
        - w4 * dropout_rate
    )
    return reward, curr_acc


class FedAvgWithDQN(FedAvg):

    def __init__(self, dqn_agent: DQNAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent    = dqn_agent
        self.k_select = dqn_agent.k_select

        self._client_state = default_client_state(N_CLIENTS)
        self._prev_state   = self._client_state.flatten()
        self._prev_action  = list(range(self.k_select))
        self._prev_acc     = 0.0

        self.history_metrics: list[dict] = []

    def configure_fit(self, server_round, parameters, client_manager):
        all_clients = list(client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        ))

        # ── [BUG-1 FIX] CID 기준 정렬 ──────────────────────────────
        # client_manager.sample()은 라운드마다 순서가 달라질 수 있음.
        # all_clients[i]가 매 라운드 동일한 클라이언트를 가리키도록 cid로 정렬.
        all_clients.sort(key=lambda c: int(c.cid))
        n = len(all_clients)

        # ── [BUG-1 FIX] 중복 제거 + k_select 미달 보충 ─────────────
        raw_idx = self.agent.get_action(self._prev_state)
        # 모듈로 처리 후 dict.fromkeys로 순서 유지하며 중복 제거
        seen = {}
        for i in raw_idx:
            seen[i % n] = None
        selected_idx = list(seen.keys())

        # k_select에 못 미치면 미선택 클라이언트 중 랜덤 보충
        if len(selected_idx) < self.k_select:
            remaining = [i for i in range(n) if i not in seen]
            need = min(self.k_select - len(selected_idx), len(remaining))
            selected_idx += random.sample(remaining, need)

        selected_clients = [all_clients[i] for i in selected_idx]
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

        # ── [BUG-2 FIX] cid 누락/범위 초과 시 경고 로그 ────────────
        for m in metrics_list:
            cid = m.get("cid")
            if cid is None:
                print(f"[경고] cid 없는 메트릭 수신: {m}")
                continue
            if not (0 <= cid < N_CLIENTS):
                print(f"[경고] cid 범위 초과 (cid={cid}), 무시합니다.")
                continue
            self._client_state[cid] = normalize_row(m)

        next_state = self._client_state.flatten()

        reward, curr_acc = compute_reward(
            metrics_list, dropout_count, self._prev_acc, self.k_select
        )

        self.agent.append_sample(self._prev_state, self._prev_action, reward, next_state, False)
        loss = self.agent.train_step()

        avg_he      = float(np.mean([m.get("he_latency", 0.5) for m in metrics_list]))
        avg_he_norm = float(np.mean([
            np.clip(m.get("he_latency", 0.5) / HE_MAX, 0.0, 1.0)
            for m in metrics_list
        ]))
        avg_data = float(np.mean([m.get("data_size", 500) for m in metrics_list]))

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
            f"he={avg_he:.3f}s | data={avg_data:.0f} | "
            f"reward={reward:.4f} | "
            f"loss={f'{loss:.4f}' if loss else 'buffer 부족'}"
        )

        self._prev_acc   = curr_acc
        self._prev_state = next_state

        return aggregated_params, aggregated_metrics