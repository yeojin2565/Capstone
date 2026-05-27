"""
compare_results_w3.py

DQN w3 sensitivity analysis 비교 그래프
w3 ∈ {0.3, 0.4, 0.55, 0.7, 0.9}

사용법:
    python compare_results_w3.py 
        --w03_path  outputs/YYYY-MM-DD/HH-MM-SS/results_03.pkl 
        --w04_path  outputs/YYYY-MM-DD/HH-MM-SS/results_04.pkl 
        --w055_path outputs/YYYY-MM-DD/HH-MM-SS/results.pkl 
        --w07_path  outputs/YYYY-MM-DD/HH-MM-SS/results_07.pkl
        --w09_path  outputs/YYYY-MM-DD/HH-MM-SS/results_09.pkl 
        --save_dir  results/w3_comparison
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ── 색상  ──────────────────────────────────────────────────────────
W3_CONFIGS = {
    0.30: {"color": "#0098BE", "label": "w3=0.30"},
    0.40: {"color": "#130FF1", "label": "w3=0.40"},
    0.55: {"color": "#1D9E75", "label": "w3=0.55 (default)"},
    0.70: {"color": "#FFCD70", "label": "w3=0.70"},
    0.90: {"color": "#E74C3C", "label": "w3=0.90"},
}



# ── 유틸 ──────────────────────────────────────────────────────────
def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_history(history):
    rounds_loss, losses = [], []
    if history.losses_centralized:
        rounds_loss, losses = zip(*history.losses_centralized)
        rounds_loss, losses = list(rounds_loss), list(losses)

    rounds_acc, accuracies = [], []
    if "accuracy" in history.metrics_centralized:
        rounds_acc, accuracies = zip(*history.metrics_centralized["accuracy"])
        rounds_acc, accuracies = list(rounds_acc), list(accuracies)

    return rounds_loss, losses, rounds_acc, accuracies


def add_moving_average(ax, x, y, color, window: int = 5):
    if len(y) < 2:
        return
    x, y   = list(x), list(y)
    window = min(window, len(y))
    kernel = np.ones(window) / window
    ma     = np.convolve(y, kernel, mode="valid")
    x_ma   = x[window - 1: window - 1 + len(ma)]
    ax.plot(x_ma, ma, color=color, linewidth=2.2, linestyle="-", alpha=0.85,
            label="_nolegend_")


def legend_handles(w3_keys):
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0],
               color=W3_CONFIGS[w3]["color"],
               linewidth=2,
               label=W3_CONFIGS[w3]["label"])
        for w3 in w3_keys
    ]


# ── 메인 비교 그래프 (Accuracy / HE Latency / Reward) ────────────
def plot_comparison(results: dict, save_dir: str = "."):
    """
    results = {
        0.30: pkl_dict,
        0.40: pkl_dict,
        0.55: pkl_dict,
        0.70: pkl_dict,
        0.90: pkl_dict,
    }
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "DQN w3 Sensitivity Analysis",
        fontsize=14, fontweight="bold", y=1.01,
    )

    w3_keys = sorted(results.keys())

    for w3, data in sorted(results.items()):
        cfg     = W3_CONFIGS[w3]
        color   = cfg["color"]
        metrics = data.get("dqn_metrics", [])

        _, _, rounds_acc, acc = parse_history(data["history"])
        rounds_he = [m["round"]               for m in metrics]
        he_norms  = [m["avg_he_latency_norm"]  for m in metrics]
        rewards   = [m["reward"]               for m in metrics]

        # ── Accuracy ──────────────────────────────────────────
        ax = axes[0]
        if acc:
            ax.plot(rounds_acc, acc, color=color, linewidth=1.2, alpha=0.30)
            add_moving_average(ax, rounds_acc, acc, color)

        # ── HE Latency ────────────────────────────────────────
        ax = axes[1]
        if he_norms:
            ax.plot(rounds_he, he_norms, color=color, linewidth=1.2, alpha=0.30)
            add_moving_average(ax, rounds_he, he_norms, color)

        # ── Reward ────────────────────────────────────────────
        ax = axes[2]
        if rewards:
            ax.plot(rounds_he, rewards, color=color, linewidth=1.2, alpha=0.30)
            add_moving_average(ax, rounds_he, rewards, color)

    # ── 축 꾸미기 ─────────────────────────────────────────────
    handles = legend_handles(w3_keys)

    axes[0].set_title("Global Accuracy",              fontsize=12)
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(handles=handles, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[1].set_title("Avg HE Latency (normalized)",  fontsize=12)
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("HE Latency (norm)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(handles=handles, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[2].set_title("Reward per Round",              fontsize=12)
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("Reward")
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[2].legend(handles=handles, fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = Path(save_dir) / "w3_comparison.png"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"비교 그래프 저장 완료: {out}")


# ── Pareto Curve ──────────────────────────────────────────────────
def plot_pareto(results: dict, save_dir: str = "."):
    """
    x축: 평균 HE latency (초)
    y축: 최종 accuracy
    각 점에 w3 레이블 표시
    """
    points = []
    for w3, data in sorted(results.items()):
        metrics  = data.get("dqn_metrics", [])
        _, _, _, acc = parse_history(data["history"])

        final_acc = acc[-1]           if acc     else 0.0
        avg_he    = np.mean([m["avg_he_latency"] for m in metrics]) if metrics else 0.0
        points.append((w3, final_acc, avg_he))

    w3s   = [p[0] for p in points]
    accs  = [p[1] for p in points]
    hes   = [p[2] for p in points]
    colors = [W3_CONFIGS[w3]["color"] for w3 in w3s]

    fig, ax = plt.subplots(figsize=(7, 5))

    # 연결선
    ax.plot(hes, accs, color="gray", linestyle="--", linewidth=1.0,
            alpha=0.5, zorder=3)

    # 점
    for w3, acc, he, color in zip(w3s, accs, hes, colors):
        ax.scatter(he, acc, color=color, s=140, zorder=5,
                   label=W3_CONFIGS[w3]["label"], edgecolors="white", linewidths=0.8)
        ax.annotate(
            f"w3={w3}",
            (he, acc),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color=color,
        )

    ax.set_xlabel("Avg HE Latency (sec)", fontsize=11)
    ax.set_ylabel("Final Accuracy",       fontsize=11)
    ax.set_title("Accuracy–HE Latency Pareto Curve\n(DQN w3 Sensitivity)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path(save_dir) / "pareto_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Pareto curve 저장 완료: {out}")


# ── 요약 테이블 출력 ──────────────────────────────────────────────
def print_summary(results: dict):
    print("\n" + "─" * 65)
    print(f"{'w3':>6} {'최종 Acc':>10} {'최고 Acc':>10} "
          f"{'평균 HE(s)':>12} {'평균 HE(norm)':>14} {'평균 Reward':>12}")
    print("─" * 65)

    for w3, data in sorted(results.items()):
        metrics      = data.get("dqn_metrics", [])
        _, _, _, acc = parse_history(data["history"])

        final_acc  = acc[-1]  if acc     else 0.0
        best_acc   = max(acc) if acc     else 0.0
        avg_he     = np.mean([m["avg_he_latency"]      for m in metrics]) if metrics else 0.0
        avg_he_n   = np.mean([m["avg_he_latency_norm"] for m in metrics]) if metrics else 0.0
        avg_reward = np.mean([m["reward"]              for m in metrics]) if metrics else 0.0

        print(f"{w3:>6.2f} {final_acc:>10.4f} {best_acc:>10.4f} "
              f"{avg_he:>12.4f} {avg_he_n:>14.4f} {avg_reward:>12.4f}")

    print("─" * 65)


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w03_path",  type=str, required=True)
    parser.add_argument("--w04_path",  type=str, required=True)
    parser.add_argument("--w055_path", type=str, required=True)
    parser.add_argument("--w07_path",  type=str, required=True)
    parser.add_argument("--w09_path",  type=str, required=True)
    parser.add_argument("--save_dir",  type=str, default="outputs/w3_comparison")
    args = parser.parse_args()

    results = {
        0.30: load(args.w03_path),
        0.40: load(args.w04_path),
        0.55: load(args.w055_path),
        0.70: load(args.w07_path),
        0.90: load(args.w09_path),
    }

    plot_comparison(results, save_dir=args.save_dir)
    plot_pareto(results,     save_dir=args.save_dir)
    print_summary(results)