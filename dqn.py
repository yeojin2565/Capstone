"""
dqn.py

DQN 에이전트 - 클라이언트별 독립 스코어링 (5피처 버전)

[변경 이력]
    [BUG-6]  fit_pca() 미호출 시 RuntimeError → no-op으로 처리
    [BUG-7]  미사용 import 제거
    [ARCH]   FC 네트워크 → Per-client ScoringNetwork
             이유: FC는 모든 입력을 섞어 "client i의 HE → Q[i]" 학습 불가
                  Per-client는 클라이언트별 독립 처리 → 단순하고 빠른 수렴
    [FEAT]   N_CLIENT_FEATURES 2 → 5 (원래 5개 피처 복원)
             [loss_norm, acc_norm, train_latency_norm, he_latency_norm, data_size_norm]
             Per-client 구조 덕분에 5개 피처도 FC처럼 섞이지 않고
             클라이언트별로 독립 처리 → 학습 가능
    [TUNE]   epsilon 감소 train_step()에서 제거
             → dqn_strategy.py aggregate_fit()에서 라운드 기반 처리
    [TUNE]   EPSILON_DECAY=0.95, BATCH_SIZE=32, MEMORY_SIZE=5000

타임라인 (200라운드):
    Round  1~32 : 메모리 축적, epsilon=1.0
    Round 33~91 : 학습 + epsilon 0.95씩 감소 → 0.05
    Round 92~200: 109라운드 exploitation
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Hyperparameters ─────────────────────────────────────
N_CLIENTS         = 100
N_CLIENT_FEATURES = 5              # loss, accuracy, train_latency, he_latency, data_size
STATE_SIZE        = N_CLIENTS * N_CLIENT_FEATURES   # 500
K_SELECT          = 10

GAMMA         = 0.95
LR            = 0.0003
EPSILON_START = 1.0
EPSILON_DECAY = 0.95
EPSILON_MIN   = 0.05
BATCH_SIZE    = 32
MEMORY_SIZE   = 5000
TARGET_UPDATE = 15


class ScoringNetwork(nn.Module):
    """
    클라이언트별 독립 스코어링 네트워크
    입력: [loss_norm, acc_norm, train_lat_norm, he_norm, data_norm]  (5차원)
    출력: score (1차원)

    동일 가중치를 100개 클라이언트에 공유 적용 (weight sharing)
    → 클라이언트 i의 스코어는 오직 클라이언트 i의 5개 피처만으로 결정
    → 네트워크가 학습하는 것:
        "he_norm 낮고 data_norm 높고 acc_norm 높은 클라이언트 = 높은 score"
    """
    def __init__(self, n_features: int = N_CLIENT_FEATURES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n_clients, n_features]
        반환: [batch, n_clients]
        """
        return self.fc(x).squeeze(-1)


class DQNAgent:
    def __init__(
        self,
        state_size:   int = STATE_SIZE,   # backward compatibility
        n_clients:    int = N_CLIENTS,
        k_select:     int = K_SELECT,
        n_components: int = 0,            # 미사용 (backward compatibility)
    ):
        self.n_clients  = n_clients
        self.k_select   = k_select
        self.state_size = n_clients * N_CLIENT_FEATURES   # 항상 500
        self.epsilon    = EPSILON_START
        self.memory     = deque(maxlen=MEMORY_SIZE)
        self.step_count = 0

        self.model        = ScoringNetwork(N_CLIENT_FEATURES)
        self.target_model = ScoringNetwork(N_CLIENT_FEATURES)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-4)

    # ── PCA (no-op) ─────────────────────────────────────
    def fit_pca(self, states: np.ndarray = None):
        """Per-client scoring 사용으로 PCA 불필요. train_dqn.py 호환용."""
        print("[DQN] Per-client scoring(5피처) 사용 중. fit_pca() 무시됨.")

    def _reshape(self, state: np.ndarray) -> np.ndarray:
        """flat [n_clients * 5] → [n_clients, 5]"""
        return state.flatten()[:self.state_size].reshape(
            self.n_clients, N_CLIENT_FEATURES
        ).astype(np.float32)

    # ── Action ─────────────────────────────────────────
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state: np.ndarray) -> list[int]:
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.n_clients), self.k_select)

        s_t = torch.FloatTensor(self._reshape(state)).unsqueeze(0)  # [1, 100, 5]
        with torch.no_grad():
            scores = self.model(s_t).squeeze(0)   # [100]
        return torch.topk(scores, self.k_select).indices.tolist()

    # ── Memory ─────────────────────────────────────────
    def append_sample(self, state, action, reward, next_state, done):
        s  = np.array(state,      dtype=np.float32).flatten()[:self.state_size]
        ns = np.array(next_state, dtype=np.float32).flatten()[:self.state_size]

        action_vec = np.zeros(self.n_clients, dtype=np.float32)
        for idx in action:
            action_vec[idx] = 1.0

        self.memory.append((s, action_vec, float(reward), ns, float(done)))

    # ── Train ──────────────────────────────────────────
    def train_step(self):
        # epsilon이 MIN에 도달 = Q값이 수렴한 시점
        # 이후 계속 학습하면 드리프트 발생 → 학습 중단
        if self.epsilon <= EPSILON_MIN:
            return None
        
        if len(self.memory) < BATCH_SIZE:
            return None

        batch  = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        B = BATCH_SIZE
        states_t      = torch.FloatTensor(np.array(states)).view(B, self.n_clients, N_CLIENT_FEATURES)
        actions_t     = torch.FloatTensor(np.array(actions))
        rewards_t     = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states)).view(B, self.n_clients, N_CLIENT_FEATURES)
        dones_t       = torch.FloatTensor(dones)

        curr_scores = self.model(states_t)                          # [B, 100]
        curr_q      = (curr_scores * actions_t).sum(1) / self.k_select

        with torch.no_grad():
            next_scores = self.target_model(next_states_t)          # [B, 100]
            next_q      = next_scores.max(1)[0]
            target_q    = rewards_t + (1 - dones_t) * GAMMA * next_q

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.update_target_model()

        return loss.item()

    # ── Save / Load ────────────────────────────────────
    def save(self, path: str):
        torch.save({"model_state": self.model.state_dict()}, path)

    def load(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["model_state"])