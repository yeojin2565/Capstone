"""
dqn_strategy.py

FedAvg 상속 → DQN 기반 클라이언트 선택

흐름:
    Round r 종료
        → aggregate_fit()에서 클라이언트 metrics 수집
        → next_state 구성 + reward 계산
        → DQN transition 저장 & train_step
    Round r+1 시작
        → configure_fit()에서 DQN이 클라이언트 선택
"""

import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes, Parameters

from dqn import DQNAgent, K_SELECT, N_CLIENTS, N_FEATURES, STATE_SIZE

# ── 정규화 기준값 ──────────────────────────────────────
SCALE = np.array([5.0, 1.0, 10.0, 6.0, 2000.0], dtype=np.float32)
# loss, accuracy, train_latency, he_latency(max=6.0), data_size


def default_state(n_clients: int = N_CLIENTS) -> np.ndarray:
    """첫 라운드용 중립 state"""
    rows = np.tile([1.0, 0.5, 0.5, 0.5, 0.5], (n_clients, 1)).astype(np.float32)
    return rows.flatten()


def normalize_metrics(metrics_list: list[dict]) -> np.ndarray:
    """metrics dict 리스트 → 정규화된 flatten 벡터"""
    rows = []
    for m in metrics_list:
        row = np.array([
            m.get("loss",          1.0),
            m.get("accuracy",      0.5),
            m.get("train_latency", 0.5),
            m.get("he_latency",    0.5),
            m.get("data_size",     1000),
        ], dtype=np.float32)
        rows.append(np.clip(row / SCALE, 0.0, 1.0))
    return np.array(rows, dtype=np.float32).flatten()


def compute_reward(
    metrics_list: list[dict],
    dropout_count: int,
    alpha: float = 1.0,
    beta: float  = 0.3,
) -> float:
    """
    R = -avg_he_latency_norm + alpha * avg_accuracy - beta * dropout_count

    - HE latency 낮은 클라이언트 선택 → reward 증가
    - accuracy 높은 클라이언트 선택  → reward 증가 (보조)
    - dropout 발생                  → reward 감소
    """
    he_norms  = [np.clip(m.get("he_latency", 0.5) / 6.0, 0.0, 1.0) for m in metrics_list if m]
    accs      = [m.get("accuracy", 0.0) for m in metrics_list if m]

    avg_he  = float(np.mean(he_norms)) if he_norms else 0.5
    avg_acc = float(np.mean(accs))     if accs     else 0.0

    return -avg_he + alpha * avg_acc - beta * dropout_count


class FedAvgWithDQN(FedAvg):

    def __init__(self, dqn_agent: DQNAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = dqn_agent

        self._prev_state  = default_state()
        self._prev_action = list(range(K_SELECT))
        self.history_metrics: list[dict] = []

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

        selected_idx = self.agent.get_action(self._prev_state)
        # 범위 초과 방지
        selected_idx     = [i % len(all_clients) for i in selected_idx]
        selected_clients = [all_clients[i] for i in selected_idx]
        self._prev_action = selected_idx

        print(
            f"\n[DQN] Round {server_round} | "
            f"selected={selected_idx} | epsilon={self.agent.epsilon:.3f}"
        )

        config  = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)
        return [(c, fit_ins) for c in selected_clients]

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

        # metrics 수집
        metrics_list = [fit_res.metrics or {} for _, fit_res in results]

        # dropped 클라이언트 수 (전송 노이즈 dropout)
        dropout_count = sum(m.get("dropped", 0) for m in metrics_list) + len(failures)

        # next_state 구성 (N_CLIENTS 수 맞추기)
        padded = metrics_list + [{}] * (N_CLIENTS - len(metrics_list))
        next_state = normalize_metrics(padded[:N_CLIENTS])

        # reward 계산
        reward = compute_reward(metrics_list, dropout_count)

        # DQN 업데이트
        self.agent.append_sample(self._prev_state, self._prev_action, reward, next_state, False)
        loss = self.agent.train_step()

        # 기록
        curr_acc = float(np.mean([m.get("accuracy", 0.0) for m in metrics_list]))
        avg_he   = float(np.mean([m.get("he_latency", 0.5) for m in metrics_list]))

        self.history_metrics.append({
            "round":         server_round,
            "accuracy":      curr_acc,
            "avg_he_latency": avg_he,
            "reward":        reward,
            "dropout_count": dropout_count,
            "dqn_loss":      loss,
            "epsilon": self.agent.epsilon
        })

        print(
            f"[DQN] Round {server_round} 완료 | "
            f"acc={curr_acc:.4f} | he={avg_he:.3f}s | "
            f"reward={reward:.4f} | "
            f"loss={f'{loss:.4f}' if loss else 'buffer 부족'}"
        )

        self._prev_state = next_state
        return aggregated_params, aggregated_metrics