# edge_server.py
from crypto import decrypt_weights
import random
import numpy as np
from model import DQNAgent

# ── Reward 계산 ───────────────────────────────────────
MAX_HE_LATENCY = 1.0   # normalize_metrics에서 [0, 1] 정규화했으므로

def compute_reward(avg_he_latency_norm, beta=0.3, dropout_count=0):
    # type: (float, float, int) -> float
    """
    R = -(avg_he_latency_norm) - beta * dropout

    - HE latency 절감이 DQN의 핵심 학습 목표
    - he_latency 낮은 클라이언트 선택 -> reward 증가
    - he_latency 높은 클라이언트 선택 -> reward 감소
    """
    return -avg_he_latency_norm - beta * dropout_count


class EdgeServer:

    def __init__(self, clients, state_size, n_clients, k_select):
        self.clients = clients
        self.last_selected_clients = []

        self.agent = DQNAgent(state_size, n_clients, k_select)

        self.prev_state  = None
        self.prev_action = None

        self.round_count = 0

        # 라운드별 metrics 기록 (실험 결과용)
        self.history_metrics = []  # type: list

    def make_state(self):
        # 정규화 기준값: loss, accuracy, train_latency, he_latency, data_size
        # data_size가 수백~수천 단위라 그대로 넣으면 state scale이 폭발함 → 정규화 필수
        scale = np.array([2.0, 1.0, 800.0, 1.0, 3000.0], dtype=np.float32)

        state = []
        for client in self.clients:
            s = client.get_state()
            if s is None:
                # 첫 라운드 전 기본값: 중립적인 초기 state
                row = np.array([1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            else:
                row = np.array([
                    s["loss"],
                    s["accuracy"],
                    s["train_latency"],
                    s["he_latency"],
                    s["data_size"]
                ], dtype=np.float32)
                row = np.clip(row / scale, 0.0, 1.0)
            state.append(row)
        return np.array(state, dtype=np.float32).flatten()

    def aggregate_encrypted(self, results):
        agg = None
        for weights, _ in results:
            if agg is None:
                agg = weights
            else:
                for i in range(len(agg)):
                    for j in range(len(agg[i])):
                        agg[i][j] += weights[i][j]
        return agg

    def fit(self, global_weights, config):
        results = []

        k      = config["clients_per_edge_per_round"]
        warmup = config.get("warmup_rounds", 0)

        state = self.make_state()

        # -------------------------------
        # 클라이언트 선택
        # -------------------------------
        if self.round_count < warmup:
            selected_ids   = random.sample(range(len(self.clients)), k)
            selection_type = "Random"
        else:
            selected_ids   = self.agent.get_action(state)
            selection_type = "RL"

        selected_clients = [self.clients[i] for i in selected_ids]
        self.last_selected_clients = selected_clients

        # ---- 선택된 클라이언트 출력 ----
        print(f"[Round {self.round_count+1} | {selection_type}] Selected clients: {selected_ids}")

        # 클라이언트별 학습
        for client in selected_clients:
            weights, data_size, _ = client.fit(global_weights, config)
            results.append((weights, data_size))

        # 암호화된 가중치 합산 후 복호화
        enc_sum = self.aggregate_encrypted(results)
        shapes  = [w.shape for w in global_weights]
        dec_sum = decrypt_weights(enc_sum, shapes)

        total_clients = len(results)
        new_weights   = [w / total_clients for w in dec_sum]

        # ── RL reward 계산 (신규: HE latency 기반) ─────────────────
        # he_latency 정규화 후 평균으로 reward 산출
        he_latencies_norm = []
        for client in selected_clients:
            s = client.get_state()
            if s is not None:
                he_norm = float(np.clip(s["he_latency"] / MAX_HE_LATENCY, 0.0, 1.0))
                he_latencies_norm.append(he_norm)

        avg_he_latency_norm = float(np.mean(he_latencies_norm)) if he_latencies_norm else 0.5

        reward = compute_reward(
            avg_he_latency_norm=avg_he_latency_norm,
            dropout_count=0,   # edge 내부에서는 dropout 없음
        )

        next_state = self.make_state()

        if self.prev_state is not None:
            self.agent.append_sample(
                self.prev_state,
                self.prev_action,
                reward,
                next_state,
                False
            )
            dqn_loss = self.agent.train_step()
        else:
            dqn_loss = None

        # ── accuracy 추출 (기록용) ───────────────────
        accs = [
            client.get_state()["accuracy"]
            for client in selected_clients
            if client.get_state() is not None
        ]
        curr_accuracy = float(np.mean(accs)) if accs else 0.0

        # ── 라운드별 기록 저장 ─────────────────────────
        self.history_metrics.append({
            "round":               self.round_count + 1,
            "accuracy":            curr_accuracy,
            "avg_he_latency_norm": avg_he_latency_norm,
            "reward":              reward,
            "dqn_loss":            dqn_loss,
            "epsilon":             self.agent.epsilon,
        })

        self.prev_state  = state
        self.prev_action = selected_ids
        self.round_count += 1

        return new_weights, total_clients

    def get_states(self):
        states = []
        for client in self.last_selected_clients:
            state = client.get_state()
            if state is not None:
                states.append(state)
        return states
