"""
dqn_strategy.py
 
FedAvg를 상속받아 configure_fit()을 오버라이드.
DQN 에이전트가 이전 라운드 클라이언트 metrics를 보고
다음 라운드에 참여할 클라이언트를 선택한다.
 
흐름:
    Round r 종료
        → aggregate_fit()에서 각 클라이언트 metrics 수집
        → DQN state 구성
        → Round r+1 시작
        → configure_fit()에서 DQN이 클라이언트 선택
"""

import numpy as np
from logging import WARNING
from typing import Optional
 
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
 
from DQN import DQNAgent, K_SELECT

# ── 상수 ──────────────────────────────────────────────
N_CLIENTS  = 30   # 엣지당 클라이언트 수 
N_FEATURES = 5    # loss, accuracy, train_latency, he_latency, data_size
STATE_SIZE = N_CLIENTS * N_FEATURES  
 
 
# ── 기본 state (첫 라운드용) ───────────────────────────
def default_state(n_clients: int = N_CLIENTS) -> np.ndarray:
    """
    첫 라운드는 이전 metrics가 없으므로
    중립적인 초기 state로 대체
    """
    rows = np.tile(
        [1.0, 0.5, 0.5, 0.5, 0.5],  # loss, acc, train_lat, he_lat, data_size (정규화값)
        (n_clients, 1)
    ).astype(np.float32)
    return rows.flatten()
 
 
# ── State 정규화 ──────────────────────────────────────
def normalize_metrics(metrics_list: list[dict]) -> np.ndarray:
    """
    클라이언트 metrics dict 리스트 → 정규화된 (N_CLIENTS × N_FEATURES) flatten 벡터
 
    metrics 키: loss, accuracy, train_latency, he_latency, data_size
    """
    # 정규화 기준값 
    scale = np.array([2.0, 1.0, 800.0, 1.0, 3000.0], dtype=np.float32)
 
    rows = []
    for m in metrics_list:
        row = np.array([
            m.get("loss",          1.0),
            m.get("accuracy",      0.5),
            m.get("train_latency", 0.5),
            m.get("he_latency",    0.5),
            m.get("data_size",     1000),
        ], dtype=np.float32)
        row = np.clip(row / scale, 0.0, 1.0)
        rows.append(row)
 
    return np.array(rows, dtype=np.float32).flatten()  # (30*5,)
 
 
# ── Reward 계산 ───────────────────────────────────────
MAX_HE_LATENCY = 1.0   # normalize_matrics에서 [0, 1] 정규화했으므로

def compute_reward(
    avg_he_latency_norm: float,
    beta: float = 0.3,
    dropout_count: int = 0,
) -> float:
    """
    R = -(avg_he_latency_norm) - beta * dropout
    
    - HE latency 절감이 DQN의 핵심 학습 목표
    - he_latency 낮은 클라이언트 선택 -> reward 증가
    - he_latency 높은 클라이언트 선택 -> reward 감소
    """
    return -avg_he_latency_norm - beta * dropout_count
 
 
# ── DQN 커스텀 전략 ───────────────────────────────────
class FedAvgWithDQN(FedAvg):
    """
    FedAvg를 상속.
    configure_fit()  : DQN이 선택한 클라이언트에게만 fit 지시
    aggregate_fit()  : 부모 집계 후 DQN 업데이트용 transition 저장
    """
 
    def __init__(self, dqn_agent: DQNAgent, he_context=None, **kwargs):
        super().__init__(**kwargs)
        self.agent = dqn_agent
        self.he_context = he_context  # 비밀키 포함 컨텍스트 (복호화용)
 
        # 라운드 간 전달용 버퍼
        self._prev_state    = default_state(N_CLIENTS)
        self._prev_action   = list(range(K_SELECT))   # 초기 더미
        self._prev_accuracy = 0.0
        self._prev_time     = 0.0
        self._round_start   = None
 
        # 라운드별 metrics 기록 (실험 결과용)
        self.history_metrics: list[dict] = []
 
    # ── 1. 학습 참여 클라이언트 선택 ──────────────────
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        DQN이 현재 state를 보고 K개 클라이언트 선택.
        첫 라운드는 default_state 사용.
        """
        import time
        self._round_start = time.time()
 
        # 전체 사용 가능한 클라이언트 목록
        all_clients = list(
            client_manager.sample(
                num_clients=client_manager.num_available(),
                min_num_clients=client_manager.num_available(),
            )
        )
 
        # DQN 선택 (인덱스 기준)
        available_n = min(len(all_clients), N_CLIENTS)
        state = self._prev_state
        selected_idx = self.agent.get_action(state)
 
        # 인덱스가 available_n 범위 초과 방지
        selected_idx = [i % available_n for i in selected_idx]
        selected_clients = [all_clients[i] for i in selected_idx]
 
        self._prev_action = selected_idx
 
        print(
            f"\n[DQN] Round {server_round} | "
            f"선택된 클라이언트 인덱스: {selected_idx} | "
            f"epsilon: {self.agent.epsilon:.3f}"
        )
 
        # FitIns 구성 (config는 부모 on_fit_config_fn 활용)
        config = {}
        if self.on_fit_config_fn:
            config = self.on_fit_config_fn(server_round)
 
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in selected_clients]
 
    # ── 2. 집계 + DQN 업데이트 ────────────────────────
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ):
        """
        부모 FedAvg 집계 수행 후:
          - 클라이언트 metrics로 next_state 구성
          - reward 계산
          - DQN에 transition 저장 & train_step
        """
        import time
 
        # 부모 집계 (평문 FedAvg fallback용)
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
 
        if not results:
            return aggregated_params, aggregated_metrics
        
        if self.he_context is not None:
            from server import aggregate_he_weights
            he_weights = aggregate_he_weights(results, self.he_context)
            if he_weights is not None:
                # HE 복호화된 가중치로 교체
                aggregated_params = ndarrays_to_parameters(he_weights)
                print(f"[DQN] Round {server_round} HE FedAvg 적용 완료")
            else:
                print(f"[DQN] Round {server_round} HE 실패 → 평문 FedAvg 사용")
 
        # ── metrics 수집 ──────────────────────────────
        metrics_list = []
        for _, fit_res in results:
            m = fit_res.metrics or {}
            metrics_list.append(m)
 
        # N_CLIENTS 수 맞추기 (결과가 적으면 패딩)
        while len(metrics_list) < N_CLIENTS:
            metrics_list.append({})
 
        next_state = normalize_metrics(metrics_list[:N_CLIENTS])
 
        # ── accuracy 추출 (기록용) ───────────────────
        accuracies = [m.get("accuracy", 0.0) for m in metrics_list]
        curr_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
 
        # ── round duration ─────────────────────────────
        round_duration = time.time() - self._round_start if self._round_start else 1.0
        dropout_count  = len(failures)

        # ── he_latency 추출 ────────────────────────────
        # next_state에서 he_latency 위치: feature 순서상 index 3
        # (loss=0, accuracy=1, train_latency=2, he_latency=3, data_size=4)
        he_latencies_norm = [
            float(np.clip(m.get("he_latency", 0.5) / 1.0, 0.0, 1.0))
            for m in metrics_list if m
        ]
        avg_he_latency_norm = float(np.mean(he_latencies_norm)) if he_latencies_norm else 0.5
 
        dropout_count = len(failures)
 
        reward = compute_reward(
            avg_he_latency_norm=avg_he_latency_norm,
            dropout_count=dropout_count,
        )
 
        # ── DQN transition 저장 & 학습 ─────────────────
        done = False  # FL은 에피소드가 길기 때문에 중간에 done=True 없음
        self.agent.append_sample(
            self._prev_state,
            self._prev_action,
            reward,
            next_state,
            done,
        )
        loss = self.agent.train_step()
 
        # ── 기록 ──────────────────────────────────────
        self.history_metrics.append({
            "round":               server_round,
            "accuracy":            curr_accuracy,
            "avg_he_latency_norm": avg_he_latency_norm,
            "reward":              reward,
            "round_duration":      round_duration,
            "dropout_count":       dropout_count,
            "dqn_loss":            loss,
            "epsilon":             self.agent.epsilon
        })
 
        print(
            f"[DQN] Round {server_round} 집계 완료 | "
            f"accuracy={curr_accuracy:.4f} | "
            f"avg_he_latency(norm)={avg_he_latency_norm:.4f} | "
            f"reward={reward:.4f} | "
            f"dqn_loss={f'{loss:.4f}' if loss is not None else 'buffer 부족'}"
        )
 
        # 다음 라운드를 위해 state 업데이트
        self._prev_state    = next_state
        self._prev_accuracy = curr_accuracy
 
        return aggregated_params, aggregated_metrics