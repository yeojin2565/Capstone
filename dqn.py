"""
dqn.py

DQN 에이전트 (PCA 차원 축소 포함)

수정 사항:
    [BUG-6 경미] _transform(): fit_pca() 미호출 시 raw(180차원)가 24차원 QNetwork에 들어가
                 크래시 발생. 명시적 RuntimeError로 즉시 알려주도록 수정.
    [BUG-7 경미] save(): 미사용 import pickle, pathlib 제거.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.decomposition import PCA

# ── Hyperparameters ─────────────────────────────────────
N_CLIENTS     = 36
N_FEATURES    = 5
STATE_SIZE    = N_CLIENTS * N_FEATURES   # 180 (raw)
N_COMPONENTS  = 24                       # PCA 축소 후 차원 (180 → 24)
K_SELECT      = 10

GAMMA         = 0.95
LR            = 0.001
EPSILON_START = 1.0
EPSILON_DECAY = 0.97
EPSILON_MIN   = 0.05
BATCH_SIZE    = 32
MEMORY_SIZE   = 2000
TARGET_UPDATE = 10


class QNetwork(nn.Module):
    def __init__(self, input_size: int, n_clients: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_clients),
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int = STATE_SIZE,
        n_clients:  int = N_CLIENTS,
        k_select:   int = K_SELECT,
        n_components: int = N_COMPONENTS,
    ):
        self.state_size   = state_size
        self.n_clients    = n_clients
        self.k_select     = k_select
        self.n_components = n_components
        self.epsilon      = EPSILON_START
        self.memory       = deque(maxlen=MEMORY_SIZE)
        self.step_count   = 0

        self.pca: PCA | None = None

        # QNetwork 입력 크기 = PCA 축소 차원.
        # fit_pca()를 반드시 먼저 호출해야 함.
        net_input = n_components
        self.model        = QNetwork(net_input, n_clients)
        self.target_model = QNetwork(net_input, n_clients)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    # ── PCA ────────────────────────────────────────────
    def fit_pca(self, states: np.ndarray):
        """
        states: [N, STATE_SIZE] — 초기 state 샘플로 PCA 학습
        학습 시작 전 1회 반드시 호출할 것.
        """
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(states)
        explained = self.pca.explained_variance_ratio_.sum()
        print(
            f"[PCA] {self.n_components}개 컴포넌트 | "
            f"분산 설명률: {explained:.1%}"
        )

    def _transform(self, state: np.ndarray) -> np.ndarray:
        """raw state(180차원) → PCA 축소 state(n_components차원)"""
        # ── [BUG-6 FIX] fit_pca() 미호출 시 즉시 명시적 에러 ────────
        # 수정 전: raw 180차원을 그대로 반환 → QNetwork(입력=24) forward에서 크래시
        # 수정 후: 사용자에게 fit_pca() 호출 누락임을 명확히 알림
        if self.pca is None:
            raise RuntimeError(
                "[DQNAgent] fit_pca()를 먼저 호출하세요. "
                f"현재 state 크기({self.state_size})와 "
                f"QNetwork 입력 크기({self.n_components})가 불일치합니다."
            )
        return self.pca.transform(
            state.reshape(1, -1)
        ).flatten().astype(np.float32)

    # ── Action ─────────────────────────────────────────
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state: np.ndarray) -> list[int]:
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.n_clients), self.k_select)

        state_t = torch.FloatTensor(self._transform(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t).squeeze(0)
        return torch.topk(q_values, self.k_select).indices.tolist()

    # ── Memory ─────────────────────────────────────────
    def append_sample(self, state, action, reward, next_state, done):
        s  = self._transform(np.array(state,      dtype=np.float32))
        ns = self._transform(np.array(next_state, dtype=np.float32))

        action_vec = np.zeros(self.n_clients, dtype=np.float32)
        for idx in action:
            action_vec[idx] = 1.0

        self.memory.append((s, action_vec, reward, ns, done))

    # ── Train ──────────────────────────────────────────
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

        curr_q = (self.model(states) * actions).sum(1) / self.k_select

        with torch.no_grad():
            next_q   = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.update_target_model()

        return loss.item()

    # ── Save / Load ────────────────────────────────────
    def save(self, path: str):
        # ── [BUG-7 FIX] 미사용 import pickle, pathlib 제거 ──────────
        data = {
            "model_state": self.model.state_dict(),
            "pca":         self.pca,
        }
        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["model_state"])
        self.pca = data.get("pca", None)