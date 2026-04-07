"""
plot_results.py

results.pkl에서 FL 학습 결과를 불러와
Global Loss / Accuracy / Convergence 그래프를 그린다.

사용법:
    python plot_results.py --results_path outputs/YYYY-MM-DD/HH-MM-SS/results.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ── 결과 로드 ──────────────────────────────────────────
def load_results(results_path: str) -> dict:
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    return results


# ── history 파싱 ───────────────────────────────────────
def parse_history(history):
    """
    flwr History 객체에서 round별 loss / accuracy 추출

    history.losses_centralized  : [(round, loss), ...]
    history.metrics_centralized : {"accuracy": [(round, acc), ...]}
    """
    rounds_loss, losses = zip(*history.losses_centralized) \
        if history.losses_centralized else ([], [])

    accuracies, rounds_acc = [], []
    if "accuracy" in history.metrics_centralized:
        rounds_acc, accuracies = zip(*history.metrics_centralized["accuracy"])

    return (
        list(rounds_loss), list(losses),
        list(rounds_acc),  list(accuracies),
    )


# ── Convergence 계산 ───────────────────────────────────
def find_convergence_round(accuracies: list, threshold: float = 0.90) -> int | None:
    """
    accuracy가 threshold를 처음 넘은 라운드 반환
    못 넘으면 None
    """
    for i, acc in enumerate(accuracies):
        if acc >= threshold:
            return i + 1
    return None


# ── 그래프 그리기 ──────────────────────────────────────
def plot_results(results: dict, save_dir: str = "."):
    history      = results["history"]
    dqn_metrics  = results.get("dqn_metrics", [])   # 없으면 빈 리스트

    rounds_loss, losses, rounds_acc, accuracies = parse_history(history)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── 색상 팔레트 ────────────────────────────────────
    COLOR_LOSS  = "#D85A30"
    COLOR_ACC   = "#1D9E75"
    COLOR_CONV  = "#534AB7"
    COLOR_REW   = "#BA7517"
    COLOR_SHADE = "#E0E0E0"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Federated Learning with DQN Client Selection", fontsize=14, fontweight="bold", y=0.98)

    # ── 1. Global Loss ─────────────────────────────────
    ax1 = axes[0][0]
    if losses:
        ax1.plot(rounds_loss, losses, color=COLOR_LOSS, linewidth=2, marker="o", markersize=4)
        ax1.fill_between(rounds_loss, losses, alpha=0.08, color=COLOR_LOSS)
    ax1.set_title("Global Loss", fontsize=12)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3)

    # ── 2. Global Accuracy ─────────────────────────────
    ax2 = axes[0][1]
    CONV_THRESHOLD = 0.90
    conv_round = find_convergence_round(accuracies, threshold=CONV_THRESHOLD)

    if accuracies:
        ax2.plot(rounds_acc, accuracies, color=COLOR_ACC, linewidth=2, marker="o", markersize=4)
        ax2.fill_between(rounds_acc, accuracies, alpha=0.08, color=COLOR_ACC)
        ax2.axhline(CONV_THRESHOLD, color=COLOR_CONV, linestyle="--",
                    linewidth=1.2, label=f"threshold ({CONV_THRESHOLD:.0%})")
        if conv_round:
            ax2.axvline(conv_round, color=COLOR_CONV, linestyle=":",
                        linewidth=1.2, label=f"수렴 라운드: {conv_round}")
        ax2.legend(fontsize=9)

    ax2.set_title("Global Accuracy", fontsize=12)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.grid(True, alpha=0.3)

    # ── 3. Convergence (accuracy 변화량) ────────────────
    ax3 = axes[1][0]
    if len(accuracies) > 1:
        delta_acc = np.diff(accuracies).tolist()
        delta_rounds = rounds_acc[1:]
        colors = [COLOR_ACC if d >= 0 else COLOR_LOSS for d in delta_acc]
        ax3.bar(delta_rounds, delta_acc, color=colors, alpha=0.75)
        ax3.axhline(0, color="gray", linewidth=0.8)

    ax3.set_title("Accuracy Δ per Round (convergence speed)", fontsize=12)
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Δ Accuracy")
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.grid(True, alpha=0.3, axis="y")

    # ── 4. DQN Reward (있을 때만) ──────────────────────
    ax4 = axes[1][1]
    if dqn_metrics:
        dqn_rounds  = [m["round"]   for m in dqn_metrics]
        dqn_rewards = [m["reward"]  for m in dqn_metrics]
        dqn_losses  = [m["dqn_loss"] for m in dqn_metrics if m["dqn_loss"] is not None]

        ax4.plot(dqn_rounds, dqn_rewards, color=COLOR_REW, linewidth=2,
                 marker="s", markersize=4, label="reward")
        ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax4.set_title("DQN Reward per Round", fontsize=12)
        ax4.set_xlabel("Round")
        ax4.set_ylabel("Reward")
        ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, "DQN metrics 없음\n(dqn_metrics 키 확인)",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=11, color="gray")
        ax4.set_title("DQN Reward per Round", fontsize=12)

    ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    out_file = save_path / "fl_results.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"그래프 저장 완료: {out_file}")

    # ── 수치 요약 출력 ────────────────────────────────
    print("\n── 결과 요약 ──")
    if losses:
        print(f"최종 Loss    : {losses[-1]:.4f}")
    if accuracies:
        print(f"최종 Accuracy: {accuracies[-1]:.4f}")
        print(f"최고 Accuracy: {max(accuracies):.4f} (Round {rounds_acc[np.argmax(accuracies)]})")
    if conv_round:
        print(f"수렴 라운드  : Round {conv_round} (threshold={CONV_THRESHOLD:.0%})")
    else:
        print(f"수렴 미달    : 전체 {len(accuracies)} 라운드 내 {CONV_THRESHOLD:.0%} 미도달")
    if dqn_metrics:
        avg_reward = np.mean([m["reward"] for m in dqn_metrics])
        print(f"DQN 평균 Reward: {avg_reward:.4f}")


# ── 메인 ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        default="results.pkl",
        help="results.pkl 경로",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="그래프 저장 디렉토리",
    )
    args = parser.parse_args()

    results = load_results(args.results_path)
    plot_results(results, save_dir=args.save_dir)