import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random
from stable_baselines3 import A2C

# [1. 결과가 매번 똑같이 나오도록 고정]
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# [2. 기본 설정값]
N_CLIENTS = 10
N_FEATURES = 5  # 5가지 정보 (정확도, loss, 시간 등)
STATE_SIZE = N_CLIENTS * N_FEATURES # 10명 x 5개 = 50차원 상황판
TOTAL_TIMESTEPS = 1000

# [3. AI가 연습할 가상 연습장 정의]
class FLClientEnv(gym.Env):
    """
    기기들의 정보를 보고 어떤 기기를 뽑을지 결정해보는 연습장.
    """
    def __init__(self, n_clients=N_CLIENTS, n_features=N_FEATURES):
        super(FLClientEnv, self).__init__()
        self.n_clients = n_clients
        self.n_features = n_features
        
        # Action: 10명 중 누구를 뽑을지 정하는 0과 1의 리스트
        self.action_space = spaces.MultiBinary(n_clients)
        
        # Observation: 우리 규격에 맞춘 50개의 숫자 상황판
        self.observation_space = spaces.Box(
            low=-10, high=1000, shape=(n_clients * n_features,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """라운드를 새로 시작할 때 상황판을 초기화합니다."""
        super().reset(seed=seed)
        initial_state = np.random.rand(self.n_clients * self.n_features).astype(np.float32)
        return initial_state, {}

    def step(self, action, y_labels=None):
        """
        AI가 기기를 선택했을 때 우리 방식대로 점수(보상)를 계산합니다.
        좋은 기기를 많이 뽑으면 플러스 점수, 나쁜 기기를 뽑으면 마이너스 점수입니다.
        """
        selected_indices = np.where(action == 1)[0]
        selected_count = len(selected_indices)
        
        # 여진이의 보상 계산 로직을 그대로 가져옴.
        if y_labels is not None and selected_count > 0:
            selected_labels = y_labels[selected_indices]
            n_good = selected_labels.sum()  # 좋은 기기(1) 개수
            n_bad = selected_count - n_good # 나쁜 기기(0) 개수
            reward = (n_good - n_bad) / selected_count # 여진 수식 반영
        else:
            reward = 0.0
            
        # 다음 라운드 상황판과 종료 여부 설정
        next_state = np.random.rand(self.n_clients * self.n_features).astype(np.float32)
        done = False
        truncated = False
        
        return next_state, reward, done, truncated, {}

# [4. 판단을 내리는 AI 본체 (A2C)]
class EdgeServerAgent:
    """
    여진님의 요구사항을 모두 맞춘 AI 에이전트입니다.
    """
    def __init__(self, n_clients=N_CLIENTS, n_features=N_FEATURES):
        self.env = FLClientEnv(n_clients, n_features)
        # A2C 방식을 사용합니다.
        self.model = A2C("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps=TOTAL_TIMESTEPS):
        """AI에게 반복해서 연습을 시키는 함수입니다."""
        print(f"학습 시작... (총 {total_timesteps}번 시뮬레이션)")
        self.model.learn(total_timesteps=total_timesteps)

    def select_clients(self, observation):
        """상황판을 보고 여진님이 원하는 [1, 0, 0...1] 형태의 결과를 ID로 바꿔줍니다."""
        action, _ = self.model.predict(observation)
        selected_ids = np.where(action == 1)[0]
        return selected_ids.tolist()

# [5. 실제로 잘 돌아가는지 확인하는 코드]
if __name__ == "__main__":
    # 에이전트 생성 및 1000번 연습 시작
    agent = EdgeServerAgent(n_clients=N_CLIENTS, n_features=N_FEATURES)
    agent.train(total_timesteps=TOTAL_TIMESTEPS)
    
    
    test_obs = np.random.rand(50).astype(np.float32)
    result = agent.select_clients(test_obs)
    
    print("-" * 50)
    print(f"테스트 결과 - 선택된 기기 ID: {result}")
    print("-" * 50)
    print("AI가 여진님 데이터 규격에 맞춰 정상 동작하고 있습니다.")