"""
rl/a2c.py

Shared Encoder + A2C 에이전트 (Permutation Invariant)

DQN 대비 핵심 차이:
    - Replay memory 없음 → 매 라운드 on-policy 업데이트
    - Actor: 클라이언트별 선택 확률 출력
    - Critic: 현재 state value 추정
    - 탐색: entropy 보너스 (epsilon-greedy 대신)

아키텍처:
    x_k ∈ R^N_FEATURES       : k번째 클라이언트 feature
    e_k = f(x_k)              : Shared Encoder → embedding
    c   = mean(e_1,...,e_N)   : Global context (전체 클라이언트 평균)
    logit_k = g(e_k || c)     : 클라이언트별 스코어 (feature + context)
    V(s)    = h(c)            : State value

Permutation invariant:
    클라이언트 순서(위치)가 아닌 feature로 구분
    → 클라이언트 추가/제거에 robust
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── 상수 ──────────────────────────────────────────────
N_CLIENTS  = 36
N_FEATURES = 6     # he_latency, accuracy, loss, train_latency, data_size, recent_dropout_rate
EMBED_DIM  = 16
K_SELECT   = 10

LR          = 0.0003
GAMMA       = 0.95
ENTROPY_COEF = 0.05   # 탐색 강도 (클수록 다양한 클라이언트 선택)
VALUE_COEF   = 0.5    # Critic loss 가중치


# ── Shared Encoder ─────────────────────────────────────
class SharedEncoder(nn.Module):
    """
    모든 클라이언트에 동일한 가중치 적용
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
    Shared Encoder + Global Context → Per-Client Scoring (Permutation Invariant)

    기존 구조의 문제:
        flatten(e_1,...,e_N) → backbone → actor
        → 클라이언트 위치(index)로 구분 → position bias 발생

    개선된 구조:
        e_k = encoder(x_k)              : 클라이언트별 embedding
        context = mean(e_1,...,e_N)     : 전체 상태 요약
        logit_k = actor(e_k || context) : feature + 전체 맥락으로 각 클라이언트 스코어링

    효과:
        - Permutation invariant: 클라이언트 순서 바뀌어도 결과 동일
        - 클라이언트를 위치가 아닌 feature로 구분
        - "다른 클라이언트와 비교해서 이 클라이언트가 얼마나 좋은가" 학습 가능

    forward:
        x       : (batch, N_CLIENTS * N_FEATURES)
        e       : (batch, N_CLIENTS, EMBED_DIM)
        context : (batch, N_CLIENTS, EMBED_DIM)  - 전체 평균 broadcast
        e_cat   : (batch, N_CLIENTS, EMBED_DIM*2)
        logits  : (batch, N_CLIENTS)
        value   : (batch, 1)
    """
    def __init__(self, n_features=N_FEATURES, n_clients=N_CLIENTS, embed_dim=EMBED_DIM):
        super().__init__()
        self.n_clients = n_clients
        self.embed_dim = embed_dim

        self.encoder = SharedEncoder(n_features, embed_dim)

        # 각 클라이언트: 자신의 embedding + global context → 스코어 1개
        self.actor = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Critic: global context → state value
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]

        # ── Shared Encoder: 클라이언트별 독립 처리 ────────
        x_flat = x.view(batch * self.n_clients, -1)
        e      = self.encoder(x_flat)                         # (batch*N, embed_dim)
        e      = e.view(batch, self.n_clients, self.embed_dim) # (batch, N, embed_dim)

        # ── Global Context: 전체 클라이언트 평균 ──────────
        # "현재 라운드 전체 클라이언트의 상태 요약"
        context = e.mean(dim=1, keepdim=True)                  # (batch, 1, embed_dim)
        context = context.expand(-1, self.n_clients, -1)       # (batch, N, embed_dim)

        # ── Per-Client Scoring ────────────────────────────
        # 각 클라이언트: 자신의 feature + 전체 context 비교
        e_cat  = torch.cat([e, context], dim=-1)               # (batch, N, embed_dim*2)
        logits = self.actor(e_cat).squeeze(-1)                 # (batch, N)

        # ── State Value ───────────────────────────────────
        value  = self.critic(context[:, 0, :])                 # (batch, 1)

        return logits, value


# ── A2C 에이전트 ──────────────────────────────────────
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
        매 라운드 종료 후 A2C 업데이트 (replay memory 불필요)

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