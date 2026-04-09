# DQN.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ----- Hyperparameters -----
GAMMA           = 0.95
LR              = 0.001
EPSILON_START   = 1.0
EPSILON_DECAY   = 0.97
EPSILON_MIN     = 0.01
BATCH_SIZE      = 32
MEMORY_SIZE     = 2000
N_CLIENTS       = 30    # 엣지당 클라이언트 수
K_SELECT        = 10     # 매 라운드 선택할 클라이언트 수
TARGET_UPDATE   = 10    # 몇 스텝마다 target network 동기화


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),  # 각 클라이언트의 Q값
        )
        
    def forward(self, x):
        return self.fc(x)
    
    
class DQNAgent:
    def __init__(self, state_size, n_clients=N_CLIENTS, k_select=K_SELECT):
        self.state_size = state_size
        self.action_size = n_clients   # 클라이언트 수 = action 차원
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
        
    def get_action(self, state) -> list[int]:
        """
        K개 클라이언트 인덱스 리스트 반환
        
        Epsilon-greedy:
          - 탐색(random): 클라이언트 중 K개 무작위 선택
          - 활용(greedy): Q값 상위 K개 선택
        """
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size), self.k_select)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, state_size)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)    # (n_clients,)
            
        # Q값 상위 K개 인덱스 선택
        top_k = torch.topk(q_values, self.k_select).indices.tolist()
        return top_k
    
    def append_sample(self, state, action, reward, next_state, done):
        """
        action: 선택된 클라이언트 인덱스 리스트 (예: [0, 3, 7])
        → 저장 시 multi-hot 벡터로 변환 (예: [1,0,0,1,0,0,0,1,0,0])
        """

        # FIXME: 클라이언트 개수에 따라 action vector 차원 바뀌어야함
        action_vec = np.zeros(self.action_size, dtype=np.float32)
        for idx in action:
            action_vec[idx] = 1.0
        self.memory.append((state, action_vec, reward, next_state, done))
        
    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        batch = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))       # (B, state_size)
        actions     = torch.FloatTensor(np.array(actions))      # (B, n_clients)
        rewards     = torch.FloatTensor(rewards)                 # (B,)
        next_states = torch.FloatTensor(np.array(next_states))  # (B, state_size)
        dones       = torch.FloatTensor(dones)                   # (B,)

        # 현재 Q값: 선택된 클라이언트들의 Q값 평균
        curr_q_all = self.model(states)                          # (B, n_clients)
        curr_q     = (curr_q_all * actions).sum(1) / self.k_select  # (B,)

        # 타겟 Q값 (Bellman)
        with torch.no_grad():
            next_q  = self.target_model(next_states).max(1)[0]  # (B,)
            target_q = rewards + (1 - dones) * GAMMA * next_q   # (B,)

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
            self.epsilon  = max(self.epsilon, EPSILON_MIN)

        # 주기적 target network 동기화
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

    N_FEATURES = 5   # loss, accuracy, train_latency, he_latency, data_size
    state_size = N_CLIENTS * N_FEATURES  # 10 × 5 = 50

    agent = DQNAgent(state_size=state_size, n_clients=N_CLIENTS, k_select=K_SELECT)

    print("── DQN 단독 테스트 ──")
    for round_idx in range(20):
        state      = np.random.rand(state_size).astype(np.float32)
        next_state = np.random.rand(state_size).astype(np.float32)

        selected = agent.get_action(state)
        reward   = random.uniform(-1.0, 1.0)
        done     = False

        agent.append_sample(state, selected, reward, next_state, done)
        loss = agent.train_step()

        print(
            f"Round {round_idx+1:02d} | "
            f"selected={selected} | "
            f"reward={reward:.3f} | "
            f"epsilon={agent.epsilon:.3f} | "
            f"loss={loss if loss else 'buffer 부족'}"
        )
