"""
dqn.py

Shared Encoder + DQN 에이전트
 
아키텍처:
    x_k ∈ R^N_FEATURES  : k번째 클라이언트 feature 벡터
    e_k = f(x_k)         : Shared Encoder → embedding
    E   = [e_1,...,e_N]  : 전체 클라이언트 embedding
    Q   = g(flatten(E))  : Q-network → 각 클라이언트 Q값
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Hyperparameters ─────────────────────────────────────
N_CLIENTS     = 36
N_FEATURES    = 6       # loss, accuracy, train_latency, he_latency, data_size, recent_dropout_rate
EMBED_DIM     = 16      # client embedding 차원
STATE_SIZE    = N_CLIENTS * N_FEATURES   # 216
K_SELECT      = 4      # 매 라운드 선택할 클라이언트 수

GAMMA         = 0.95
LR            = 0.001
EPSILON_START = 1.0
EPSILON_DECAY = 0.97    
EPSILON_MIN   = 0.05
BATCH_SIZE    = 32
MEMORY_SIZE   = 2000
TARGET_UPDATE = 10      # target network 동기화 주기


# ── Shared Encoder ─────────────────────────────────────
class SharedEncoder(nn.Module):
    """
    클라이언트별 feature -> embedding
    
    input : (batch * n_clients, n_features)
    output: (batch * n_clieats, embed_dim)
    """
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    
# ── Q-Network ──────────────────────────────────────────
class QNetworkWithEncoder(nn.Module):
    """
    Shared Encoder -> Q-network
    
    forward 흐름:
        x     : (batch, n_clients * n_features)
        reshape: (batch * n_clients, n_features)
        encoder: (batch * n_clients, embed_dim)
        reshape: (batch, n_clients * embed_dim)
        q_net : (batch, n_clients)
    """
    
    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_clients:  int = N_CLIENTS,
        embed_dim:  int = EMBED_DIM,
    ):
        super().__init__()
        self.n_clients = n_clients
        self.embed_dim = embed_dim
 
        self.encoder = SharedEncoder(n_features, embed_dim)
 
        self.q_net = nn.Sequential(
            nn.Linear(n_clients * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_clients),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
 
        # (batch, n_clients * n_features) → (batch * n_clients, n_features)
        x = x.view(batch * self.n_clients, -1)
 
        # shared encoder
        e = self.encoder(x)  # (batch * n_clients, embed_dim)
 
        # (batch * n_clients, embed_dim) → (batch, n_clients * embed_dim)
        e = e.view(batch, self.n_clients * self.embed_dim)
 
        return self.q_net(e)  # (batch, n_clients)
 
 
# ── DQN 에이전트 ──────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        state_size: int = STATE_SIZE,
        n_clients:  int = N_CLIENTS,
        k_select:   int = K_SELECT,
    ):
        self.state_size = state_size
        self.n_clients  = n_clients
        self.k_select   = k_select
        self.epsilon    = EPSILON_START
        self.memory     = deque(maxlen=MEMORY_SIZE)
        self.step_count = 0
 
        self.model        = QNetworkWithEncoder(N_FEATURES, n_clients, EMBED_DIM)
        self.target_model = QNetworkWithEncoder(N_FEATURES, n_clients, EMBED_DIM)
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
        """action: 클라이언트 인덱스 리스트 → multi-hot 벡터로 저장"""
        action_vec = np.zeros(self.n_clients, dtype=np.float32)
        for idx in action:
            action_vec[idx] = 1.0
        self.memory.append((state, action_vec, reward, next_state, done))
 
    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None
 
        # Reward 절댓값 기반 가중 샘플링 (중요한 경험 더 자주 학습)
        rewards = np.array([abs(exp[2]) + 1e-6 for exp in self.memory])
        probs   = rewards / rewards.sum()
        indices = np.random.choice(len(self.memory), BATCH_SIZE,
                                   replace=False, p=probs)
        batch   = [self.memory[i] for i in indices]
 
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