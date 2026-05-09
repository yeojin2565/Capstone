"""
dqn_strategy.py

FedAvg 상속 → Shared Encoder DQN 기반 클라이언트 선택

변경사항:
    - N_FEATURES = 6 (recent_dropout_rate 추가)
    - dropout 기록 관리 (window=5)
    - 선택 로그 CSV 저장
    - configure_fit에서 recent_dropout_rate를 config로 클라이언트에 전달
"""

import csv
import numpy as np
from pathlib import Path

from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes, Parameters

from dqn import DQNAgent, K_SELECT, N_CLIENTS, N_FEATURES, STATE_SIZE
from he_simulator import get_group

# ── 정규화 기준값 ──────────────────────────────────────
# [he_latency, accuracy, loss, train_latency, data_size, recent_dropout_rate]
SCALE = np.array([1.5, 1.0, 5.0, 10.0, 2000.0, 1.0], dtype=np.float32)

# dropout 기록 윈도우
DROPOUT_WINDOW = 5


def default_state(n_clients: int = N_CLIENTS) -> np.ndarray:
    """첫 라운드용 중립 state"""
    rows = np.tile(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.0],  # he, acc, loss, tlat, dsize, dropout
        (n_clients, 1)
    ).astype(np.float32)
    return rows.flatten()


def normalize_metrics(metrics_list: list[dict]) -> np.ndarray:
    """
    metrics dict 리스트 → 정규화된 flatten 벡터
    shape: (N_CLIENTS * N_FEATURES,)
    """
    rows = []
    for m in metrics_list:
        row = np.array([
            m.get("he_latency",          0.5),
            m.get("accuracy",            0.5),
            m.get("loss",                1.0),
            m.get("train_latency",       0.5),
            m.get("data_size",           1000),
            m.get("recent_dropout_rate", 0.0),
        ], dtype=np.float32)
        rows.append(np.clip(row / SCALE, 0.0, 1.0))
    return np.array(rows, dtype=np.float32).flatten()


def compute_reward(
    metrics_list: list[dict],
    dropout_count: int,
    alpha: float = 0.3,
    beta:  float = 0.3,
) -> float:
    """
    R = -avg_he_latency_norm + alpha * avg_accuracy - beta * dropout_count

    he_latency 정규화 기준: 1.5 (Slow 그룹 평균값)
    alpha 낮게 설정 → he_latency 항이 reward 지배
    """
    he_norms = [
        float(np.clip(m.get("he_latency", 0.5) / 1.5, 0.0, 1.0))
        for m in metrics_list if m
    ]
    accs = [m.get("accuracy", 0.0) for m in metrics_list if m]

    avg_he  = float(np.mean(he_norms)) if he_norms else 0.5
    avg_acc = float(np.mean(accs))     if accs     else 0.0

    return -avg_he + alpha * avg_acc - beta * dropout_count


class FedAvgWithDQN(FedAvg):

    def __init__(self, dqn_agent: DQNAgent, log_dir: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.agent    = dqn_agent
        self.log_dir  = Path(log_dir)

        self._prev_state  = default_state()
        self._prev_action = list(range(K_SELECT))

        # dropout 기록: {cid: [0, 1, 0, ...]}
        self.dropout_history: dict[int, list] = {
            cid: [] for cid in range(N_CLIENTS)
        }

        # 로그
        self.history_metrics: list[dict] = []
        self.selection_log:   list[dict] = []

    def _get_dropout_rate(self, cid: int) -> float:
        """최근 DROPOUT_WINDOW 라운드 dropout 비율"""
        hist = self.dropout_history.get(cid, [])
        return float(np.mean(hist)) if hist else 0.0

    def _update_dropout_history(self, metrics_list: list[dict]):
        """클라이언트 dropout 기록 업데이트"""
        for m in metrics_list:
            cid     = m.get("cid", -1)
            dropped = m.get("dropped", 0)
            if cid < 0:
                continue
            self.dropout_history[cid].append(dropped)
            if len(self.dropout_history[cid]) > DROPOUT_WINDOW:
                self.dropout_history[cid].pop(0)

    def _save_selection_log(self):
        """선택 로그 CSV 저장"""
        if not self.selection_log:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "selection_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "round", "epsilon", "selected_idx", "group_counts"
            ])
            writer.writeheader()
            writer.writerows(self.selection_log)

    # ── 1. 클라이언트 선택 ─────────────────────────────
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:

        all_clients  = list(client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        ))

        selected_idx     = self.agent.get_action(self._prev_state)
        selected_idx     = [i % len(all_clients) for i in selected_idx]
        selected_clients = [all_clients[i] for i in selected_idx]
        self._prev_action = selected_idx

        # 그룹별 선택 현황
        groups       = [get_group(i % len(all_clients)) for i in selected_idx]
        group_counts = {}
        for g in groups:
            group_counts[g] = group_counts.get(g, 0) + 1

        print(
            f"\n[DQN] Round {server_round} | "
            f"epsilon={self.agent.epsilon:.3f} | "
            f"groups={group_counts}"
        )

        # 로그 기록
        self.selection_log.append({
            "round":        server_round,
            "epsilon":      round(self.agent.epsilon, 4),
            "selected_idx": selected_idx,
            "group_counts": group_counts,
        })

        # recent_dropout_rate를 config로 클라이언트에 전달
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        dropout_rates = {
            i: self._get_dropout_rate(i % len(all_clients))
            for i in selected_idx
        }
        # 평균 dropout rate를 config에 추가
        config["recent_dropout_rate"] = float(np.mean(list(dropout_rates.values())))

        fit_ins = FitIns(parameters, config)
        return [(c, fit_ins) for c in selected_clients]

    # ── 2. 집계 + DQN 업데이트 ────────────────────────
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

        # dropout 기록 업데이트
        self._update_dropout_history(metrics_list)

        # dropout 수 계산
        dropout_count = sum(m.get("dropped", 0) for m in metrics_list) + len(failures)

        # recent_dropout_rate를 metrics에 반영
        for m in metrics_list:
            cid = m.get("cid", -1)
            if cid >= 0:
                m["recent_dropout_rate"] = self._get_dropout_rate(cid)

        # next_state 구성
        padded     = metrics_list + [{}] * (N_CLIENTS - len(metrics_list))
        next_state = normalize_metrics(padded[:N_CLIENTS])

        # reward 계산
        reward = compute_reward(metrics_list, dropout_count)

        # DQN 업데이트
        self.agent.append_sample(
            self._prev_state, self._prev_action, reward, next_state, False
        )
        loss = self.agent.train_step()

        # 기록
        curr_acc = float(np.mean([m.get("accuracy",   0.0) for m in metrics_list]))
        avg_he   = float(np.mean([m.get("he_latency", 0.5) for m in metrics_list]))

        self.history_metrics.append({
            "round":          server_round,
            "accuracy":       curr_acc,
            "avg_he_latency": avg_he,
            "reward":         reward,
            "dropout_count":  dropout_count,
            "epsilon":        self.agent.epsilon,
            "dqn_loss":       loss,
        })

        print(
            f"[DQN] Round {server_round} 완료 | "
            f"acc={curr_acc:.4f} | he={avg_he:.3f}s | "
            f"reward={reward:.4f} | "
            f"loss={f'{loss:.4f}' if loss else 'buffer 부족'}"
        )

        self._prev_state = next_state

        # 매 10라운드마다 선택 로그 저장
        if server_round % 10 == 0:
            self._save_selection_log()

        return aggregated_params, aggregated_metrics