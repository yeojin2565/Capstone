"""
rl/a2c.py

Shared Encoder + A2C
- Actor: 클라이언트 별 선택 확률
- Critic: 현재 state value 추정
- 탐색: entropy 보너스 (epsilon-greedy 아님)

Architecture:
    x_k ∈ R^N_FEATURES  : k번째 클라이언트 feature
    e_k = f(x_k)         : Shared Encoder → embedding
    E   = [e_1,...,e_N]  : 전체 embedding concat
    Actor  → π(a|s) : 각 클라이언트 선택 확률
    Critic → V(s)   : state value
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── Hyperparameters ────────────────────────────────────────────
N_CLIENTS  = 10
N_FEATURES = 6     # he_latency, accuracy, loss, train_latency, data_size, recent_dropout_rate
EMBED_DIM  = 16
K_SELECT   = 2

LR          = 0.0003
GAMMA       = 0.95
ENTROPY_COEF = 0.05   # 탐색 강도 (클수록 다양한 클라이언트 선택)
VALUE_COEF   = 0.5    # Critic loss 가중치


# ── Shared Encoder ─────────────────────────────────────
class SharedEncoder(nn.Module):
    """
    x_k (N_FEATURES) → e_k (EMBED_DIM)
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
    

# ── Actor-Critic Network ───────────────────────────────
class ActorCritic(nn.Module):
    """
    Shared Encoder → Actor + Critic
 
    forward:
        x      : (batch, N_CLIENTS * N_FEATURES)
        encoder: (batch * N_CLIENTS, EMBED_DIM)
        flatten: (batch, N_CLIENTS * EMBED_DIM)
        actor  : (batch, N_CLIENTS) - 선택 logits
        critic : (batch, 1)         - state value
    """
    def __init__(self, n_features=N_FEATURES, n_clients=N_CLIENTS, embed_dim=EMBED_DIM):
        super().__init__()
        self.n_clients = n_clients
        self.embed_dim = embed_dim
 
        self.encoder = SharedEncoder(n_features, embed_dim)
 
        backbone_dim = n_clients * embed_dim
 
        self.backbone = nn.Sequential(
            nn.Linear(backbone_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(64, n_clients)
        self.critic = nn.Linear(64, 1)
 
    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
 
        # Shared Encoder
        x_flat = x.view(batch * self.n_clients, -1)
        e      = self.encoder(x_flat)                    # (batch*N, embed_dim)
        e      = e.view(batch, self.n_clients * self.embed_dim)
 
        feat   = self.backbone(e)
        logits = self.actor(feat)                        # (batch, N_CLIENTS)
        value  = self.critic(feat)                       # (batch, 1)
        return logits, value
 
 
# ── A2C Agent ──────────────────────────────────────
class A2CAgent:
    def __init__(self, n_features=N_FEATURES, n_clients=N_CLIENTS, k_select=K_SELECT):
        self.n_clients = n_clients
        self.k_select  = k_select
 
        self.net       = ActorCritic(n_features, n_clients, EMBED_DIM)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
 
        # 라운드별 trajectory 저장
        self._log_probs: list[torch.Tensor] = []
        self._values:    list[torch.Tensor] = []
        self._rewards:   list[float]        = []
        self._entropies: list[torch.Tensor] = []
 
    def select_clients(self, state: np.ndarray) -> list[int]:
        """
        Actor 확률 분포에서 K개 클라이언트 비복원 샘플링
 
        탐색 방식: entropy 보너스 (epsilon-greedy 대신)
          - logits가 균등할수록 entropy 높음 → 다양한 선택
          - 학습되면서 좋은 클라이언트 logit이 높아짐 → 확률 집중
        """
        obs    = torch.FloatTensor(state).unsqueeze(0)   # (1, state_size)
        logits, value = self.net(obs)
        logits = logits.squeeze(0)                       # (N_CLIENTS,)
 
        # 비복원 샘플링: 선택한 클라이언트를 mask 처리
        selected   = []
        log_prob   = torch.zeros(1)
        mask       = torch.zeros(self.n_clients)
 
        for _ in range(self.k_select):
            masked_logits = logits - mask * 1e9          # 선택된 클라이언트 제외
            probs         = torch.softmax(masked_logits, dim=0)
            dist          = Categorical(probs)
            action        = dist.sample()
            log_prob      = log_prob + dist.log_prob(action)
            selected.append(action.item())
            mask[action.item()] = 1.0
 
        # entropy: 전체 분포의 다양성 측정
        full_probs = torch.softmax(logits, dim=0)
        entropy    = -(full_probs * torch.log(full_probs + 1e-8)).sum()
 
        self._log_probs.append(log_prob)
        self._values.append(value.squeeze())
        self._entropies.append(entropy)
 
        return selected
 
    def store_reward(self, reward: float):
        self._rewards.append(reward)
 
    def update(self) -> dict | None:
        """
        매 라운드 종료 후 A2C 업데이트
 
        advantage = R - V(s)
        actor_loss  = -log_prob * advantage
        critic_loss = (R - V(s))^2
        entropy_loss = -entropy  (탐색 장려)
        """
        if not self._rewards:
            return None
 
        # discounted return 계산
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
 
        returns   = torch.FloatTensor(returns)
        values    = torch.stack(self._values)
        log_probs = torch.stack(self._log_probs).squeeze(-1)
        entropies = torch.stack(self._entropies)
 
        # 정규화
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
 
        advantage   = (returns - values.detach())
        actor_loss  = -(log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_loss = -entropies.mean()
 
        loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss
 
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()
 
        result = {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy":     entropies.mean().item(),
            "total_loss":  loss.item(),
        }
 
        # trajectory 초기화
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._entropies.clear()
 
        return result
 
    def save(self, path: str):
        torch.save(self.net.state_dict(), path)
 
    def load(self, path: str):
        self.net.load_state_dict(torch.load(path))