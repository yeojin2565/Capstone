"""
dqn.py

DQN 에이전트
- State : 36 clients × 5 features = 180차원
- Action: Q값 상위 K개 클라이언트 인덱스 반환
- Replay memory + target network
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Hyperparameters ─────────────────────────────────────
N_CLIENTS     = 36
N_FEATURES    = 5       # loss, accuracy, train_latency, he_latency, data_size
STATE_SIZE    = N_CLIENTS * N_FEATURES   # 180
K_SELECT      = 10      # 매 라운드 선택할 클라이언트 수

GAMMA         = 0.95
LR            = 0.001
EPSILON_START = 1.0
EPSILON_DECAY = 0.97    # 100라운드 기준 ≈ 0.05 (충분한 decay)
EPSILON_MIN   = 0.05
BATCH_SIZE    = 32
MEMORY_SIZE   = 2000
TARGET_UPDATE = 10      # target network 동기화 주기


class QNetwork(nn.Module):
    def __init__(self, state_size: int, n_clients: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_clients),  # 각 클라이언트의 Q값
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int = STATE_SIZE,
        n_clients: int  = N_CLIENTS,
        k_select: int   = K_SELECT,
    ):
        self.state_size  = state_size
        self.n_clients   = n_clients
        self.k_select    = k_select
        self.epsilon     = EPSILON_START
        self.memory      = deque(maxlen=MEMORY_SIZE)
        self.step_count  = 0

        self.model        = QNetwork(state_size, n_clients)
        self.target_model = QNetwork(state_size, n_clients)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state: np.ndarray) -> list[int]:
        """
        Epsilon-greedy 클라이언트 선택
        탐색: K개 무작위 선택
        활용: Q값 상위 K개 선택
        """
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.n_clients), self.k_select)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t).squeeze(0)

        return torch.topk(q_values, self.k_select).indices.tolist()

    def append_sample(self, state, action, reward, next_state, done):
        """
        action: 선택된 클라이언트 인덱스 리스트 → multi-hot 벡터로 저장 [1, 0, 0, ...]
        """
        action_vec = np.zeros(self.n_clients, dtype=np.float32)
        for idx in action:
            action_vec[idx] = 1.0
        self.memory.append((state, action_vec, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        batch  = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.FloatTensor(np.array(actions))
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        # 선택된 클라이언트들의 Q값 평균
        curr_q = (self.model(states) * actions).sum(1) / self.k_select

        with torch.no_grad():
            next_q   = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # epsilon decay
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        # target network 동기화
        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.update_target_model()

        return loss.item()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))