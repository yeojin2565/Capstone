# edge_server.py
from crypto import decrypt_weights
import random
import numpy as np
from model import DQNAgent

class EdgeServer:

    def __init__(self, clients, state_size, n_clients, k_select):
        self.clients = clients
        self.last_selected_clients = []

        self.agent = DQNAgent(state_size, n_clients, k_select)

        self.prev_state = None
        self.prev_action = None

        self.round_count = 0

    def make_state(self):
        state = []
        for client in self.clients:
            s = client.get_state()
            if s is None:
                s = [0, 0, 0, 0, 0]
            else:
                s = [
                    s["loss"],
                    s["accuracy"],
                    s["train_latency"],
                    s["he_latency"],
                    s["data_size"]
                ]
            state.append(s)
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

        k = config["clients_per_edge_per_round"]
        warmup = config.get("warmup_rounds", 0)

        state = self.make_state()

        # -------------------------------
        # 클라이언트 선택
        # -------------------------------
        if self.round_count < warmup:
            selected_ids = random.sample(range(len(self.clients)), k)
            selection_type = "Random"
        else:
            selected_ids = self.agent.get_action(state)
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
        shapes = [w.shape for w in global_weights]
        dec_sum = decrypt_weights(enc_sum, shapes)

        total_clients = len(results)
        new_weights = [w / total_clients for w in dec_sum]

        # RL reward 계산
        accs = [client.get_state()["accuracy"] for client in selected_clients if client.get_state() is not None]
        reward = np.mean(accs) if accs else 0

        next_state = self.make_state()

        if self.prev_state is not None:
            self.agent.append_sample(
                self.prev_state,
                self.prev_action,
                reward,
                next_state,
                False
            )
            self.agent.train_step()

        self.prev_state = state
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