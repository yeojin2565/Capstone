"""
dqn.py

DQN 에이전트 - 클라이언트별 독립 스코어링 네트워크

# 클라이언트별 독립 스코어링 (Per-client Scoring)
    각 client i에 대해: [he_i_norm, data_i_norm] → score(i)
    동일한 작은 네트워크를 100개 클라이언트에 공유 적용 (weight sharing)
    → client i의 스코어는 오직 client i의 피처로만 결정
    → 네트워크가 배워야 할 것: "he_norm 낮으면 score 높게"
    → 매우 단순한 패턴 → 수십 번 학습으로 수렴

# ScoringNetwork 구조:
    입력: [he_norm, data_norm]  (2차원)
    2 → 64 → 32 → 1  (score)
    동일 네트워크를 100개 클라이언트에 동시 적용 (batched)

타임라인 (200라운드):
    Round  1~32 : 메모리 축적, epsilon=1.0
                  탐색 중 각 클라이언트의 [he_norm, data_norm] 관찰
    Round 33~91 : 학습 + epsilon 0.95씩 감소 → 0.05
                  "he_norm 낮은 클라이언트 → 높은 score" 빠르게 수렴
    Round 92~200: 109라운드 exploitation
                  fast 그룹(cid 0~34) 집중 선택 → HE latency 명확히 감소
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

    
# ── Hyperparameters ─────────────────────────────────────
# [FIXME]: 하드코딩
N_CLIENTS          = 100
N_CLIENT_FEATURES  = 2              # he_latency_norm, data_size_norm
STATE_SIZE         = N_CLIENTS * N_CLIENT_FEATURES   # 200
K_SELECT           = 10

GAMMA         = 0.95
LR            = 0.001              # 작은 네트워크에 맞게 LR 증가 (0.0005 → 0.001)
EPSILON_START = 1.0
EPSILON_DECAY = 0.95
EPSILON_MIN   = 0.05
BATCH_SIZE    = 32
MEMORY_SIZE   = 5000
TARGET_UPDATE = 5     # step(minibatch update)


class ScoringNetwork(nn.Module):
    """
    클라이언트별 독립 스코어링 네트워크
    입력: [he_norm, data_norm] (2차원)
    출력: score (1차원, 높을수록 선택 우선)

    동일 가중치를 100개 클라이언트에 공유 적용 (weight sharing).
    → 네트워크가 "특정 클라이언트를 위한 특수 룰" 이 아닌
      "어떤 피처를 가진 클라이언트가 좋은가"를 학습.
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
        반환: [batch, n_clients]  (각 클라이언트의 스코어)
        """
        return self.fc(x).squeeze(-1)


class DQNAgent:
    def __init__(
        self,
        state_size:   int = STATE_SIZE,   # backward compatibility용, 내부에서 재계산
        n_clients:    int = N_CLIENTS,
        k_select:     int = K_SELECT,
        n_components: int = 0,            # 미사용 (backward compatibility)
    ):
        self.n_clients   = n_clients
        self.k_select    = k_select
        self.state_size  = n_clients * N_CLIENT_FEATURES   # 항상 200
        self.epsilon     = EPSILON_START
        self.memory      = deque(maxlen=MEMORY_SIZE)
        self.step_count  = 0

        self.model        = ScoringNetwork(N_CLIENT_FEATURES)
        self.target_model = ScoringNetwork(N_CLIENT_FEATURES)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    # ── PCA (no-op, backward compatibility) ────────────
    def fit_pca(self, states: np.ndarray = None):
        pass
    
    def _reshape(self, state: np.ndarray) -> np.ndarray:
        """flat state [n_clients * 2] → [n_clients, 2]"""
        return state.flatten()[:self.state_size].reshape(
            self.n_clients, N_CLIENT_FEATURES
        ).astype(np.float32)

    # ── Action ─────────────────────────────────────────
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state: np.ndarray) -> list[int]:
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.n_clients), self.k_select)

        # [n_clients, 2] → [1, n_clients, 2]
        s_t = torch.FloatTensor(self._reshape(state)).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(s_t).squeeze(0)   # [n_clients]
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
        """epsilon 감소 없음 → dqn_strategy.py 라운드 기반으로 처리."""
        if len(self.memory) < BATCH_SIZE:
            return None

        batch  = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        B = BATCH_SIZE
        # [B, n_clients * 2] → [B, n_clients, 2]
        states_t      = torch.FloatTensor(np.array(states)).view(B, self.n_clients, N_CLIENT_FEATURES)
        actions_t     = torch.FloatTensor(np.array(actions))   # [B, n_clients]
        rewards_t     = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states)).view(B, self.n_clients, N_CLIENT_FEATURES)
        dones_t       = torch.FloatTensor(dones)

        # 선택된 클라이언트들의 평균 Q값 (현재)
        curr_scores = self.model(states_t)                        # [B, n_clients]
        curr_q      = (curr_scores * actions_t).sum(1) / self.k_select

        # 타깃 Q값
        with torch.no_grad():
            next_scores = self.target_model(next_states_t)        # [B, n_clients]
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