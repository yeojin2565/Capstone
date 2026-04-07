import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random
from stable_baselines3 import A2C

# [결과가 매번 똑같이 나오도록 설정]
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# [기본 설정값들]
N_CLIENTS = 10
N_FEATURES = 4  # 기기 상태 정보 개수 (배터리, 속도 등)
TOTAL_TIMESTEPS = 1000

# 1. AI가 연습할 가상 환경 만들기
class FLClientEnv(gym.Env):
    """
    기기들의 상태를 보고 어떤 기기를 뽑을지 결정하는 곳.
    """
    def __init__(self, n_clients=N_CLIENTS):
        super(FLClientEnv, self).__init__()
        self.n_clients = n_clients
        
        # Action: 10개 기기를 각각 켤지(1), 끌지(0) 정하는 스위치
        self.action_space = spaces.MultiBinary(n_clients)
        
        # Observation: 10개 기기의 정보 4개씩, 총 40개의 상황판
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(n_clients * 4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """라운드를 새로 시작할 때 기기 상태를 초기화합니다."""
        super().reset(seed=seed)
        initial_state = np.random.rand(self.n_clients * 4).astype(np.float32)
        return initial_state, {}

    def step(self, action):
        """
        AI가 기기를 선택했을 때 점수(보상)를 계산하는 곳.
        시간이 짧고, 정확도가 높고, 배터리가 많으면 높은 점수를 줍니다.
        """
        selected_indices = np.where(action == 1)[0]
        selected_count = len(selected_indices)
        
        # 기기들의 성능 데이터 가상으로 만들기
        latency = 10.0 / (selected_count + 1e-6)
        accuracy = 0.8 + (0.01 * selected_count)
        avg_battery = np.mean(np.random.rand(selected_count)) if selected_count > 0 else 0
        
        # 보상 점수 계산식
        reward = -latency + (0.5 * accuracy) + (0.2 * avg_battery)
        
        # 다음 상황과 종료 여부 전달
        next_state = np.random.rand(self.n_clients * 4).astype(np.float32)
        done = False
        truncated = False
        
        return next_state, reward, done, truncated, {}

# 2. 엣지 서버 AI 
class EdgeServerAgent:
    """
    A2C 
    """
    def __init__(self, n_clients=N_CLIENTS):
        self.env = FLClientEnv(n_clients)
        # A2C 엔진을 가져와서 환경에 연결함.
        self.model = A2C("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps=TOTAL_TIMESTEPS):
        """AI에게 연습장에서 반복 학습을 시키는 함수입니다."""
        print(f"AI 학습 진행 중... (총 {total_timesteps}번 연습)")
        self.model.learn(total_timesteps=total_timesteps)

    def select_clients(self, observation):
        """현재 상황을 보고 어떤 기기 ID를 뽑을지 리스트로 알려줍니다."""
        action, _ = self.model.predict(observation)
        selected_ids = np.where(action == 1)[0]
        return selected_ids.tolist()

# 3. 코드 테스트용
if __name__ == "__main__":
    # 에이전트 만들고 학습시키기
    agent = EdgeServerAgent(n_clients=N_CLIENTS)
    agent.train(total_timesteps=TOTAL_TIMESTEPS)
    
    # 가상의 상황을 하나 주고 잘 작동하는지 확인
    test_obs = np.random.rand(40).astype(np.float32)
    result = agent.select_clients(test_obs)
    
    print("-" * 50)
    print(f"테스트 결과 - 선택된 기기 ID: {result}")
    print("-" * 50)
    print("AI가 정상적으로 결과를 내놓고 있습니다.")