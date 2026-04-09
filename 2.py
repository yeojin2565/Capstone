# DQN_agent.py (기존 코드 스타일 유지, 수정 통합 버전)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from typing import List
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows MKL 오류 방지

# ----- Hyperparameters -----
GAMMA           = 0.95
LR              = 0.001
EPSILON_START   = 1.0
EPSILON_DECAY   = 0.97      # 기존 0.995 → 0.97
EPSILON_MIN     = 0.01
BATCH_SIZE      = 32
MEMORY_SIZE     = 2000
N_CLIENTS       = 10         # 엣지당 클라이언트 수
K_SELECT        = 3          # 매 라운드 선택할 클라이언트 수
TARGET_UPDATE   = 10         # target network 동기화 주기

# ── Q-Network ─────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.fc(x)


# ── DQN 에이전트 ───────────────────────────────────────
class DQNAgent:
    def __init__(self, state_size, n_clients=N_CLIENTS, k_select=K_SELECT):
        self.state_size = state_size
        self.action_size = n_clients
        self.k_select = k_select
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.step_count = 0

        self.epsilon = EPSILON_START

        self.model = QNetwork(state_size, self.action_size)
        self.target_model = QNetwork(state_size, self.action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state) -> List[int]:
        # Epsilon-greedy로 K개 클라이언트 선택
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size), self.k_select)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)

        top_k = torch.topk(q_values, self.k_select).indices.tolist()
        return top_k

    def append_sample(self, state, action, reward, next_state, done):
        # multi-hot vector로 변환
        action_vec = np.zeros(self.action_size, dtype=np.float32)
        for idx in action:
            action_vec[idx] = 1.0
        self.memory.append((state, action_vec, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.FloatTensor(np.array(actions))
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        # 선택된 클라이언트 Q값 평균
        curr_q_all = self.model(states)
        curr_q     = (curr_q_all * actions).sum(1) / self.k_select

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(self.epsilon, EPSILON_MIN)

        # Target network 주기적 동기화
        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.update_target_model()

        return loss.item()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))


# ── 단독 테스트 ────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    N_FEATURES = 5           # loss, accuracy, train_latency, he_latency, data_size
    state_size = N_CLIENTS * N_FEATURES

    agent = DQNAgent(state_size=state_size, n_clients=N_CLIENTS, k_select=K_SELECT)

    epsilon_history = []

    print("── DQN 단독 테스트 ──")
    for round_idx in range(100):
        state      = np.random.rand(state_size).astype(np.float32)
        next_state = np.random.rand(state_size).astype(np.float32)

        selected = agent.get_action(state)
        reward   = random.uniform(-1.0, 1.0)
        done     = False

        agent.append_sample(state, selected, reward, next_state, done)
        loss = agent.train_step()

        epsilon_history.append(agent.epsilon)

        print(
            f"Round {round_idx+1:02d} | "
            f"selected={selected} | "
            f"reward={reward:.3f} | "
            f"epsilon={agent.epsilon:.3f} | "
            f"loss={loss if loss else 'buffer 부족'}"
        )

    plt.plot(epsilon_history)
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.show()