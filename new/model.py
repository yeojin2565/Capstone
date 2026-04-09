import random
import numpy as np
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# DQN 하이퍼파라미터
# -----------------------------
GAMMA           = 0.95
LR              = 0.001
EPSILON_START   = 1.0
EPSILON_DECAY   = 0.97
EPSILON_MIN     = 0.01
BATCH_SIZE      = 32
MEMORY_SIZE     = 2000
TARGET_UPDATE   = 10    # 몇 스텝마다 target network 동기화


# -----------------------------
# DQN Q-Network 정의
# -----------------------------
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


# -----------------------------
# DQN 에이전트 정의
# -----------------------------
class DQNAgent:
    def __init__(self, state_size, n_clients, k_select):
        self.state_size  = state_size
        self.action_size = n_clients   # 클라이언트 수 = action 차원
        self.k_select    = k_select

        self.memory     = deque(maxlen=MEMORY_SIZE)
        self.step_count = 0

        self.epsilon = EPSILON_START

        # main network + target network (Double DQN 구조)
        self.model        = QNetwork(state_size, self.action_size)
        self.target_model = QNetwork(state_size, self.action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def update_target_model(self):
        # target network를 main network 가중치로 동기화
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        # type: (np.ndarray) -> List[int]
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
        curr_q_all = self.model(states)                              # (B, n_clients)
        curr_q     = (curr_q_all * actions).sum(1) / self.k_select  # (B,)

        # 타겟 Q값 (Bellman + target network)
        with torch.no_grad():
            next_q   = self.target_model(next_states).max(1)[0]  # (B,)
            target_q = rewards + (1 - dones) * GAMMA * next_q    # (B,)

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

    def save(self, path):
        # type: (str) -> None
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # type: (str) -> None
        # weights_only=False: Python 3.8 + 구버전 PyTorch 호환
        self.model.load_state_dict(torch.load(path, map_location="cpu"))


# -----------------------------
# CNN 모델 정의
# -----------------------------
class Net(nn.Module):
    def __init__(self, num_classes):
        # type: (int) -> None
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 4 * 4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -----------------------------
# CNN 학습 함수
# -----------------------------
def train(net, trainloader, optimizer, epochs, device):
    # type: (nn.Module, object, object, int, str) -> None
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


# -----------------------------
# CNN 평가 함수
# -----------------------------
def test(net, testloader, device):
    # type: (nn.Module, object, str) -> tuple
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss   += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
