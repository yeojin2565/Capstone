"""
compare_results.py

DQN 제안 방법 vs Random baseline 비교 그래프

사용법:
    python compare_results.py \
    --dqn_path outputs/YYYY-MM-DD/HH-MM-SS/results.pkl \
    --random_path outputs/YYYY-MM-DD/HH-MM-SS/results_random.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ── 색상 ──────────────────────────────────────────────
COLOR_DQN    = "#1D9E75"   # 초록  (제안 방법)
COLOR_RANDOM = "#D85A30"   # 주황  (baseline)
COLOR_SHADE  = 0.10        # fill_between 투명도


# ── 로드 ──────────────────────────────────────────────
def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── history 파싱 ───────────────────────────────────────
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


# ── 수렴 라운드 ────────────────────────────────────────
def convergence_round(accuracies: list, threshold: float) -> int | None:
    for i, a in enumerate(accuracies):
        if a >= threshold:
            return i + 1
    return None


# ── 이동 평균 추세선 ───────────────────────────────────
def add_moving_average(ax, x: list, y: list, color: str, window: int = 5):
    """
    단순 이동 평균(SMA) 추세선 추가
    window: 이동 평균 윈도우 크기 (데이터 수보다 크면 자동 축소)
    """
    if len(y) < 2:
        return
    window = min(window, len(y))
    kernel = np.ones(window) / window
    ma     = np.convolve(y, kernel, mode="valid")
    # convolve valid 모드: len(y) - window + 1 개 반환 → x 앞부분 맞추기
    x_ma   = x[window - 1:]
    ax.plot(
        x_ma, ma,
        color=color, linewidth=2.2,
        linestyle="-", alpha=0.85,
        label="_nolegend_",
    )


# ── 비교 그래프 ────────────────────────────────────────
def plot_comparison(
    dqn_results: dict,
    random_results: dict,
    save_dir: str = ".",
    conv_threshold: float = 0.90,
):
    # 데이터 파싱
    d_rl, d_loss, d_ra, d_acc = parse_history(dqn_results["history"])
    r_rl, r_loss, r_ra, r_acc = parse_history(random_results["history"])

    dqn_metrics    = dqn_results.get("dqn_metrics", [])
    random_metrics = random_results.get("dqn_metrics", [])

    dqn_he    = [m["avg_he_latency_norm"] for m in dqn_metrics]
    random_he = [m["avg_he_latency_norm"] for m in random_metrics]
    dqn_rew   = [m["reward"]              for m in dqn_metrics]
    random_rew= [m["reward"]              for m in random_metrics]

    dqn_rounds_he    = [m["round"] for m in dqn_metrics]
    random_rounds_he = [m["round"] for m in random_metrics]

    # ── 레이아웃 ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "DQN Client Selection vs Random Baseline",
        fontsize=14, fontweight="bold", y=0.98
    )

    def legend_labels():
        from matplotlib.lines import Line2D
        return [
            Line2D([0], [0], color=COLOR_DQN,    linewidth=2, label="DQN (proposed)"),
            Line2D([0], [0], color=COLOR_RANDOM,  linewidth=2, label="Random (baseline)"),
        ]

    # ── 1. Global Loss ─────────────────────────────────
    ax = axes[0][0]
    if d_loss:
        ax.plot(d_rl, d_loss, color=COLOR_DQN,   linewidth=1.5, alpha=0.35)
        ax.fill_between(d_rl, d_loss, alpha=COLOR_SHADE, color=COLOR_DQN)
        add_moving_average(ax, d_rl, d_loss, COLOR_DQN)
    if r_loss:
        ax.plot(r_rl, r_loss, color=COLOR_RANDOM, linewidth=1.5, alpha=0.35)
        ax.fill_between(r_rl, r_loss, alpha=COLOR_SHADE, color=COLOR_RANDOM)
        add_moving_average(ax, r_rl, r_loss, COLOR_RANDOM)
    ax.set_title("Global Loss", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.legend(handles=legend_labels(), fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # ── 2. Global Accuracy ─────────────────────────────
    ax = axes[0][1]
    dqn_conv    = convergence_round(d_acc, conv_threshold)
    random_conv = convergence_round(r_acc, conv_threshold)

    if d_acc:
        ax.plot(d_ra, d_acc, color=COLOR_DQN,    linewidth=1.5, alpha=0.35)
        ax.fill_between(d_ra, d_acc, alpha=COLOR_SHADE, color=COLOR_DQN)
        add_moving_average(ax, d_ra, d_acc, COLOR_DQN)
    if r_acc:
        ax.plot(r_ra, r_acc, color=COLOR_RANDOM,  linewidth=1.5, alpha=0.35)
        ax.fill_between(r_ra, r_acc, alpha=COLOR_SHADE, color=COLOR_RANDOM)
        add_moving_average(ax, r_ra, r_acc, COLOR_RANDOM)

    ax.axhline(conv_threshold, color="gray", linestyle=":", linewidth=1.2,
               label=f"threshold ({conv_threshold:.0%})")

    if dqn_conv:
        ax.axvline(dqn_conv,    color=COLOR_DQN,    linestyle=":", linewidth=1.0,
                   label=f"DQN convergence: Round {dqn_conv}")
    if random_conv:
        ax.axvline(random_conv, color=COLOR_RANDOM, linestyle=":", linewidth=1.0,
                   label=f"Random convergence: Round {random_conv}")

    ax.set_title("Global Accuracy", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # ── 3. HE Latency (핵심 비교) ──────────────────────
    ax = axes[1][0]
    if dqn_he:
        ax.plot(dqn_rounds_he,    dqn_he,    color=COLOR_DQN,    linewidth=1.5, alpha=0.35)
        ax.fill_between(dqn_rounds_he, dqn_he, alpha=COLOR_SHADE, color=COLOR_DQN)
        add_moving_average(ax, dqn_rounds_he, dqn_he, COLOR_DQN)
    if random_he:
        ax.plot(random_rounds_he, random_he, color=COLOR_RANDOM,  linewidth=1.5, alpha=0.35)
        ax.fill_between(random_rounds_he, random_he, alpha=COLOR_SHADE, color=COLOR_RANDOM)
        add_moving_average(ax, random_rounds_he, random_he, COLOR_RANDOM)

    ax.set_title("Avg HE Latency (normalized)", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("HE Latency (norm)")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=legend_labels(), fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # ── 4. Reward ──────────────────────────────────────
    ax = axes[1][1]
    if dqn_rew:
        ax.plot(dqn_rounds_he,    dqn_rew,    color=COLOR_DQN,    linewidth=1.5, alpha=0.35)
        ax.fill_between(dqn_rounds_he, dqn_rew, alpha=COLOR_SHADE, color=COLOR_DQN)
        add_moving_average(ax, dqn_rounds_he, dqn_rew, COLOR_DQN)
    if random_rew:
        ax.plot(random_rounds_he, random_rew, color=COLOR_RANDOM,  linewidth=1.5, alpha=0.35)
        ax.fill_between(random_rounds_he, random_rew, alpha=COLOR_SHADE, color=COLOR_RANDOM)
        add_moving_average(ax, random_rounds_he, random_rew, COLOR_RANDOM)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Reward per Round", fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Reward")
    ax.legend(handles=legend_labels(), fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = Path(save_dir) / "comparison.png"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"비교 그래프 저장 완료: {out}")

    # ── 수치 요약 ──────────────────────────────────────
    print("\n──────────────────────────────────────")
    print(f"{'':20s} {'DQN':>10s} {'Random':>10s}")
    print("──────────────────────────────────────")
    if d_acc and r_acc:
        print(f"{'최종 Accuracy':20s} {d_acc[-1]:>10.4f} {r_acc[-1]:>10.4f}")
        print(f"{'최고 Accuracy':20s} {max(d_acc):>10.4f} {max(r_acc):>10.4f}")
    if dqn_conv or random_conv:
        dqn_c_str    = str(dqn_conv)    if dqn_conv    else "미달"
        random_c_str = str(random_conv) if random_conv else "미달"
        print(f"{'수렴 라운드':20s} {dqn_c_str:>10s} {random_c_str:>10s}")
    if dqn_he and random_he:
        print(f"{'평균 HE Latency':20s} {np.mean(dqn_he):>10.4f} {np.mean(random_he):>10.4f}")
    if dqn_rew and random_rew:
        print(f"{'평균 Reward':20s} {np.mean(dqn_rew):>10.4f} {np.mean(random_rew):>10.4f}")
    print("──────────────────────────────────────")

# ── DQN Epsilon per Round ──────────────────────────────
def plot_epsilon_graph(dqn_results: dict, save_dir: str = "."):
    dqn_metrics = dqn_results.get("dqn_metrics", [])
    if not dqn_metrics:
        print("dqn_metrics 데이터가 없습니다.")
        return

    rounds  = [m["round"]   for m in dqn_metrics]
    epsilon = [m["epsilon"] for m in dqn_metrics]

    plt.figure(figsize=(10,5))
    plt.plot(rounds, epsilon, color=COLOR_DQN, linewidth=2)
    plt.title("DQN Epsilon per Round", fontsize=12, fontweight="bold")
    plt.xlabel("Round")
    plt.ylabel("Epsilon")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path(save_dir) / "epsilon_per_round.png"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Epsilon 그래프 저장 완료: {out}")


# ── 메인 ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn_path",    type=str, default="results_dqn.pkl")
    parser.add_argument("--random_path", type=str, default="results_random.pkl")
    parser.add_argument("--save_dir",    type=str, default=".")
    parser.add_argument("--threshold",   type=float, default=0.90)
    args = parser.parse_args()

    dqn_results    = load(args.dqn_path)
    random_results = load(args.random_path)

    plot_comparison(
        dqn_results,
        random_results,
        save_dir=args.save_dir,
        conv_threshold=args.threshold,
    )

    plot_epsilon_graph(
        dqn_results,
        save_dir=args.save_dir,
    )