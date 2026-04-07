import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from preprocessing import load_and_preprocessing, N_CLIENTS
from model import DQNAgent, K_SELECT

# ── 재현성 고정 ────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ── 상수 ──────────────────────────────────────────────
N_FEATURES  = 5                      # loss, accuracy, train_latency, he_latency, data_size
STATE_SIZE  = N_CLIENTS * N_FEATURES # 10 × 5 = 50
N_EPISODES  = 200                     # 학습 에피소드 수


# ── Reward 함수 ───────────────────────────────────────
def compute_reward(selected: list[int], y_labels: np.ndarray) -> float:
    """
    선택된 클라이언트의 레이블 기반 reward 계산
    - 선택된 클라이언트 중 label=1(좋은 클라이언트) 비율이 높을수록 양수 reward
    - label=0(나쁜 클라이언트) 선택 시 패널티

    selected  : 선택된 클라이언트 인덱스 리스트 (예: [0, 3, 7])
    y_labels  : 해당 라운드 클라이언트 10개의 레이블 배열
    """
    selected_labels = y_labels[selected]
    n_good = selected_labels.sum()               # label=1 개수
    n_bad  = len(selected) - n_good              # label=0 개수
    reward = (n_good - n_bad) / len(selected)    # -1.0 ~ 1.0
    return float(reward)


# ── 라운드 단위 state 구성 ────────────────────────────
def make_rounds(X, y, n_clients=N_CLIENTS):
    """
    X, y를 n_clients 단위로 묶어 라운드 리스트 반환
    각 라운드 = 클라이언트 n_clients 개의 state + label

    반환: [(state_vec, label_arr), ...]
      state_vec : (STATE_SIZE,) numpy array (flattened)
      label_arr : (n_clients,) numpy array
    """
    rounds = []
    n_rows = len(X)
    for start in range(0, n_rows - n_clients + 1, n_clients):
        x_chunk = X.iloc[start:start + n_clients].values.astype(np.float32)
        y_chunk = y.iloc[start:start + n_clients].values.astype(np.int32)
        state_vec = x_chunk.flatten()            # (50,)
        rounds.append((state_vec, y_chunk))
    return rounds


# ── 학습 루프 ─────────────────────────────────────────
def train(agent: DQNAgent, rounds: list, n_episodes: int = N_EPISODES):
    """
    episodes × rounds 반복 학습
    매 에피소드마다 전체 라운드를 순회
    """
    episode_rewards = []   # 에피소드별 평균 reward
    episode_losses  = []   # 에피소드별 평균 loss

    for ep in range(1, n_episodes + 1):
        ep_rewards = []
        ep_losses  = []

        for i, (state, y_labels) in enumerate(rounds):
            # 1. 행동 선택
            selected   = agent.get_action(state)

            # 2. Reward 계산
            reward     = compute_reward(selected, y_labels)

            # 3. 다음 state (다음 라운드, 없으면 현재 반복)
            next_idx   = (i + 1) % len(rounds)
            next_state = rounds[next_idx][0]
            done       = (i == len(rounds) - 1)

            # 4. 메모리 저장 & 학습
            agent.append_sample(state, selected, reward, next_state, done)
            loss = agent.train_step()

            ep_rewards.append(reward)
            if loss is not None:
                ep_losses.append(loss)

        avg_reward = np.mean(ep_rewards)
        avg_loss   = np.mean(ep_losses) if ep_losses else None
        episode_rewards.append(avg_reward)
        if avg_loss is not None:
            episode_losses.append(avg_loss)

        print(
            f"Episode {ep:03d}/{n_episodes} | "
            f"avg_reward={avg_reward:.4f} | "
            f"epsilon={agent.epsilon:.3f} | "
            f"avg_loss={f'{avg_loss:.4f}' if avg_loss is not None else 'buffer 부족'}"
        )

    return episode_rewards, episode_losses


# ── 평가 ──────────────────────────────────────────────
def evaluate(agent: DQNAgent, rounds: list):
    """
    test 데이터로 greedy 선택 (epsilon=0) 평가
    선택된 클라이언트 중 label=1 비율(precision) 출력
    """
    agent.epsilon = 0.0   # 탐색 끄고 greedy만
    precisions = []

    for state, y_labels in rounds:
        selected = agent.get_action(state)
        selected_labels = y_labels[selected]
        precision = selected_labels.sum() / len(selected)
        precisions.append(precision)

    avg_precision = np.mean(precisions)
    print(f"\n── 평가 결과 ──")
    print(f"평균 Precision (선택된 클라이언트 중 좋은 클라이언트 비율): {avg_precision:.4f}")
    return avg_precision


# ── 시각화 ────────────────────────────────────────────
def plot_results(episode_rewards: list, episode_losses: list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(episode_rewards, color="#1D9E75")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Avg Reward")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episode_losses, color="#D85A30")
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg Loss")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()
    print("그래프 저장 완료: results.png")


# ── 메인 ──────────────────────────────────────────────
if __name__ == "__main__":
    # 1. 데이터 로드 & 전처리
    print("── 데이터 로드 ──")
    X_train, X_test, y_train, y_test = load_and_preprocessing()
    print(f"train: {X_train.shape} / test: {X_test.shape}")

    # 2. 라운드 구성
    train_rounds = make_rounds(X_train, y_train)
    test_rounds  = make_rounds(X_test,  y_test)
    print(f"학습 라운드 수: {len(train_rounds)} / 테스트 라운드 수: {len(test_rounds)}")

    # 3. 에이전트 초기화
    agent = DQNAgent(state_size=STATE_SIZE, n_clients=N_CLIENTS, k_select=K_SELECT)

    # 4. 학습
    print("\n── 학습 시작 ──")
    rewards, losses = train(agent, train_rounds, n_episodes=N_EPISODES)

    # 5. 평가
    evaluate(agent, test_rounds)

    # 6. 시각화
    plot_results(rewards, losses)

    # 7. 모델 저장
    agent.save("dqn_model.pth")
    print("모델 저장 완료: dqn_model.pth")
