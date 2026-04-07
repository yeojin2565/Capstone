import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import A2C

# 1. AI 가상 환경 정의 
class FLClientEnv(gym.Env):
    def __init__(self, n_clients=10):
        super(FLClientEnv, self).__init__()
        self.n_clients = n_clients
        # Action: 10개 클라이언트 선택/미선택
        self.action_space = spaces.MultiBinary(n_clients)
        # Observation: 클라이언트 상태 정보 40개
        self.observation_space = spaces.Box(low=0, high=1000, shape=(n_clients * 4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """라운드 시작 시 초기화"""
        super().reset(seed=seed)
        initial_state = np.random.rand(self.n_clients * 4).astype(np.float32)
        return initial_state, {}

    def step(self, action):
        selected_indices = np.where(action == 1)[0]
        selected_count = len(selected_indices)
        
        # 보상 계산 (배터리 가중치 포함)
        latency = 10.0 / (selected_count + 1e-6)
        accuracy = 0.8 + (0.01 * selected_count)
        avg_battery = np.mean(np.random.rand(selected_count)) if selected_count > 0 else 0
        
        reward = -latency + (0.5 * accuracy) + (0.2 * avg_battery)
        
        # 다음 상태 생성
        next_state = np.random.rand(self.n_clients * 4).astype(np.float32)
        
        done = False
        truncated = False
        info = {}
        
        return next_state, reward, done, truncated, info

# 2. 에이전트 클래스 
class EdgeServerAgent:
    def __init__(self, n_clients=10):
        self.env = FLClientEnv(n_clients)
        # A2C 알고리즘
        self.model = A2C("MlpPolicy", self.env, verbose=1)

    
    def train(self, total_timesteps=1000):
        """AI를 학습시키는 메인 함수"""
        print(f"{total_timesteps} 단계 동안 학습을 진행합니다...")
        self.model.learn(total_timesteps=total_timesteps)

    def select_clients(self, observation):
        """ID 리스트를 반환"""
        action, _ = self.model.predict(observation)
        selected_ids = np.where(action == 1)[0]
        return selected_ids.tolist()
    
    
    # --- 여기서부터는 테스트를 위한 실행 코드 ---
if __name__ == "__main__":
    # 1. 엣지 서버 에이전트 생성
    agent = EdgeServerAgent(n_clients=10)
    
    # 2. AI 학습 시작 (1000단계 동안 스스로 공부하게 시킵니다)
    agent.train(total_timesteps=1000)
    
    # 3. 학습 결과 확인: 가상의 상태를 주고 누구를 뽑는지 물어봅니다
    test_observation = np.random.rand(40).astype(np.float32)
    selected_clients = agent.select_clients(test_observation)
    
    print("\n" + "="*50)
    print(f"테스트 결과 - 선택된 클라이언트 ID: {selected_clients}")
    print("="*50)
    print("성공! AI가 정상적으로 판단을 내리고 있습니다.")