"""
strategy/a2c_strategy.py

FedAvg 상속 → A2C 기반 클라이언트 선택

Reward 설계:
    R = -avg_he_latency_norm - beta * dropout_count
    (accuracy 항 제거 → he_latency 신호가 reward를 지배)

Normalization:
    he_latency scale = 6.0 (Extreme 최대값 기준)
    → excellent/fast/medium/slow/extreme 모두 구분 가능
"""

import csv
import numpy as np
from pathlib import Path

from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes, Parameters

from rl.a2c import A2CAgent, N_FEATURES
from src.he_simulator import get_group

# ── 정규화 기준값 ──────────────────────────────────────
# [he_latency, accuracy, loss, train_latency, data_size, recent_dropout_rate]
SCALE = np.array([6.0, 1.0, 5.0, 10.0, 2000.0, 1.0], dtype=np.float32)
#                 ^^^
# 6.0 기준: excellent=0.003, fast=0.017, medium=0.083, slow=0.25, extreme=0.833
# → 모든 그룹이 구분 가능한 값으로 정규화됨

DROPOUT_WINDOW = 5


def default_state(n_clients: int) -> np.ndarray:
    rows = np.tile([0.5, 0.5, 0.5, 0.5, 0.5, 0.0], (n_clients, 1)).astype(np.float32)
    return rows.flatten()


def normalize_metrics(metrics_list: list[dict]) -> np.ndarray:
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


def compute_reward(metrics_list: list[dict], dropout_count: int, beta: float = 0.3) -> float:
    """
    R = -avg_he_latency_norm - beta * dropout_count

    accuracy 항 제거:
        Non-IID 환경에서 slow 클라이언트가 accuracy 높을 수 있음
        → accuracy 항이 he_latency 신호를 희석시키는 문제 방지
    """
    he_norms = [
        float(np.clip(m.get("he_latency", 0.5) / 6.0, 0.0, 1.0))
        for m in metrics_list if m
    ]
    return -float(np.mean(he_norms)) - beta * dropout_count


class FedAvgWithA2C(FedAvg):

    def __init__(self, agent: A2CAgent, log_dir: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.agent   = agent
        self.log_dir = Path(log_dir)

        self._prev_state = default_state(self.agent.n_clients)
        self.dropout_history: dict[int, list] = {cid: [] for cid in range(self.agent.n_clients)}

        self.history_metrics: list[dict] = []
        self.selection_log:   list[dict] = []

    def _get_dropout_rate(self, cid: int) -> float:
        hist = self.dropout_history.get(cid, [])
        return float(np.mean(hist)) if hist else 0.0

    def _update_dropout_history(self, metrics_list: list[dict]):
        for m in metrics_list:
            cid = m.get("cid", -1)
            if cid < 0:
                continue
            self.dropout_history[cid].append(m.get("dropped", 0))
            if len(self.dropout_history[cid]) > DROPOUT_WINDOW:
                self.dropout_history[cid].pop(0)

    def _save_log(self):
        if not self.selection_log:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "selection_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "selected_idx", "group_counts"])
            writer.writeheader()
            writer.writerows(self.selection_log)

    # ── 클라이언트 선택 ────────────────────────────────
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

        # A2C가 현재 state를 보고 K개 선택
        selected_idx     = self.agent.select_clients(self._prev_state)
        selected_idx     = [i % len(all_clients) for i in selected_idx]
        selected_clients = [all_clients[i] for i in selected_idx]

        # 그룹 현황 로그
        groups       = [get_group(i % len(all_clients)) for i in selected_idx]
        group_counts = {}
        for g in groups:
            group_counts[g] = group_counts.get(g, 0) + 1

        print(f"\n[A2C] Round {server_round} | groups={group_counts}")

        self.selection_log.append({
            "round":        server_round,
            "selected_idx": selected_idx,
            "group_counts": group_counts,
        })

        # recent_dropout_rate → config로 전달
        avg_dropout = np.mean([
            self._get_dropout_rate(i % len(all_clients)) for i in selected_idx
        ])
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        config["recent_dropout_rate"] = float(avg_dropout)

        fit_ins = FitIns(parameters, config)
        return [(c, fit_ins) for c in selected_clients]

    # ── 집계 + A2C 업데이트 ───────────────────────────
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

        # recent_dropout_rate 반영
        for m in metrics_list:
            cid = m.get("cid", -1)
            if cid >= 0:
                m["recent_dropout_rate"] = self._get_dropout_rate(cid)

        dropout_count = sum(m.get("dropped", 0) for m in metrics_list) + len(failures)

        # next_state
        n = self.agent.n_clients
        padded     = metrics_list + [{}] * (n - len(metrics_list))
        next_state = normalize_metrics(padded[:n])

        # reward (accuracy 항 없음)
        reward = compute_reward(metrics_list, dropout_count)
        self.agent.store_reward(reward)

        # A2C 업데이트 (replay memory 없이 즉시)
        losses = self.agent.update()

        # 기록
        curr_acc = float(np.mean([m.get("accuracy",   0.0) for m in metrics_list]))
        avg_he   = float(np.mean([m.get("he_latency", 0.5) for m in metrics_list]))

        self.history_metrics.append({
            "round":          server_round,
            "accuracy":       curr_acc,
            "avg_he_latency": avg_he,
            "reward":         reward,
            "dropout_count":  dropout_count,
            "actor_loss":     losses["actor_loss"]  if losses else None,
            "critic_loss":    losses["critic_loss"] if losses else None,
            "entropy":        losses["entropy"]     if losses else None,
        })

        print(
            f"[A2C] Round {server_round} 완료 | "
            f"acc={curr_acc:.4f} | he={avg_he:.3f}s | reward={reward:.4f} | "
            f"entropy={losses['entropy']:.4f}" if losses else ""
        )

        self._prev_state = next_state

        if server_round % 10 == 0:
            self._save_log()

        return aggregated_params, aggregated_metrics