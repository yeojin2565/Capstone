"""
compare_results_new.py

DQN 제안 방법 vs Random baseline vs Rule-based baseline 비교 그래프

사용법:
    # 2개 비교 (기존)
    python compare_results_new.py \
        --dqn_path results.pkl \
        --random_path results_random.pkl

    # 3개 비교 (rule-based 추가)
    python compare_results_new.py \
        --dqn_path results.pkl \
        --random_path results_random.pkl \
        --rule_path results_rule_based.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from matplotlib.lines import Line2D


# ── 색상 ──────────────────────────────────────────────
COLOR_DQN       = "#1D9E75"
COLOR_RANDOM    = "#D85A30"
COLOR_RULEBASED = "#5B7FBF"
COLOR_SHADE     = 0.10


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


def convergence_round(accuracies: list, threshold: float):
    for i, a in enumerate(accuracies):
        if a >= threshold:
            return i + 1
    return None


def add_moving_average(ax, x, y, color, window=5):
    if len(y) < 2:
        return
    x, y   = list(x), list(y)
    window = min(window, len(y))
    kernel = np.ones(window) / window
    ma     = np.convolve(y, kernel, mode="valid")
    x_ma   = x[window - 1: window - 1 + len(ma)]
    ax.plot(x_ma, ma, color=color, linewidth=2.2, linestyle="-", alpha=0.85, label="_nolegend_")


def draw_series(ax, rounds, values, color):
    """반투명 원시 + 이동평균 + fill_between"""
    if not values:
        return
    ax.plot(rounds, values, color=color, linewidth=1.5, alpha=0.35)
    ax.fill_between(rounds, values, alpha=COLOR_SHADE, color=color)
    add_moving_average(ax, rounds, values, color)


def plot_comparison(
    dqn_results:    dict,
    random_results: dict,
    rule_results:   dict | None = None,
    save_dir:       str   = ".",
    conv_threshold: float = 0.90,
):
    # ── 데이터 파싱 ──────────────────────────────────────
    d_rl, d_loss, d_ra, d_acc = parse_history(dqn_results["history"])
    r_rl, r_loss, r_ra, r_acc = parse_history(random_results["history"])

    dqn_m    = dqn_results.get("dqn_metrics", [])
    random_m = random_results.get("dqn_metrics", [])
    rule_m   = rule_results.get("dqn_metrics", []) if rule_results else []

    def extract(metrics, key):
        return [m[key] for m in metrics]

    dqn_rounds    = extract(dqn_m,    "round")
    random_rounds = extract(random_m, "round")
    rule_rounds   = extract(rule_m,   "round")

    # ── 범례 핸들 ────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=COLOR_DQN,    linewidth=2, label="DQN (proposed)"),
        Line2D([0], [0], color=COLOR_RANDOM, linewidth=2, label="Random (baseline)"),
    ]
    if rule_results:
        legend_handles.append(
            Line2D([0], [0], color=COLOR_RULEBASED, linewidth=2, label="Rule-based (baseline)")
        )

    title = "DQN vs Random" + (" vs Rule-based" if rule_results else "") + " Client Selection"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # ── (0,0) Global Loss ────────────────────────────────
    ax = axes[0][0]
    draw_series(ax, d_rl, d_loss, COLOR_DQN)
    draw_series(ax, r_rl, r_loss, COLOR_RANDOM)
    if rule_results:
        rb_rl, rb_loss, _, _ = parse_history(rule_results["history"])
        draw_series(ax, rb_rl, rb_loss, COLOR_RULEBASED)
    ax.set_title("Global Loss", fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("Loss")
    ax.legend(handles=legend_handles, fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # ── (0,1) Global Accuracy ────────────────────────────
    ax = axes[0][1]
    dqn_conv    = convergence_round(d_acc, conv_threshold)
    random_conv = convergence_round(r_acc, conv_threshold)
    draw_series(ax, d_ra, d_acc, COLOR_DQN)
    draw_series(ax, r_ra, r_acc, COLOR_RANDOM)
    if rule_results:
        _, _, rb_ra, rb_acc = parse_history(rule_results["history"])
        draw_series(ax, rb_ra, rb_acc, COLOR_RULEBASED)
        rule_conv = convergence_round(rb_acc, conv_threshold)
        if rule_conv:
            ax.axvline(rule_conv, color=COLOR_RULEBASED, linestyle=":", linewidth=1.0,
                       label=f"Rule-based convergence: Round {rule_conv}")
    ax.axhline(conv_threshold, color="gray", linestyle=":", linewidth=1.2,
               label=f"threshold ({conv_threshold:.0%})")
    if dqn_conv:
        ax.axvline(dqn_conv,    color=COLOR_DQN,    linestyle=":", linewidth=1.0,
                   label=f"DQN convergence: Round {dqn_conv}")
    if random_conv:
        ax.axvline(random_conv, color=COLOR_RANDOM, linestyle=":", linewidth=1.0,
                   label=f"Random convergence: Round {random_conv}")
    ax.set_title("Global Accuracy", fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # ── (1,0) Avg HE Latency ─────────────────────────────
    ax = axes[1][0]
    draw_series(ax, dqn_rounds,    extract(dqn_m,    "avg_he_latency_norm"), COLOR_DQN)
    draw_series(ax, random_rounds, extract(random_m, "avg_he_latency_norm"), COLOR_RANDOM)
    if rule_m:
        draw_series(ax, rule_rounds, extract(rule_m, "avg_he_latency_norm"), COLOR_RULEBASED)
    ax.set_title("Avg HE Latency (normalized)", fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("HE Latency (norm)")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=legend_handles, fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # ── (1,1) Reward per Round ───────────────────────────
    ax = axes[1][1]
    draw_series(ax, dqn_rounds,    extract(dqn_m,    "reward"), COLOR_DQN)
    draw_series(ax, random_rounds, extract(random_m, "reward"), COLOR_RANDOM)
    if rule_m:
        draw_series(ax, rule_rounds, extract(rule_m, "reward"), COLOR_RULEBASED)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Reward per Round", fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("Reward")
    ax.legend(handles=legend_handles, fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / "comparison.png"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"비교 그래프 저장 완료: {out}")

    # ── 요약 테이블 ──────────────────────────────────────
    col3 = "Rule-based" if rule_results else ""
    print("\n" + "─" * 56)
    print(f"{'':20s} {'DQN':>10s} {'Random':>10s} {col3:>12s}")
    print("─" * 56)
    if rule_results:
        _, _, rb_ra, rb_acc = parse_history(rule_results["history"])
        rule_conv = convergence_round(rb_acc, conv_threshold)
    else:
        rb_acc, rule_conv = [], None

    if d_acc and r_acc:
        rb_fin = f"{rb_acc[-1]:>12.4f}" if rb_acc else f"{'N/A':>12s}"
        rb_max = f"{max(rb_acc):>12.4f}" if rb_acc else f"{'N/A':>12s}"
        print(f"{'최종 Accuracy':20s} {d_acc[-1]:>10.4f} {r_acc[-1]:>10.4f} {rb_fin}")
        print(f"{'최고 Accuracy':20s} {max(d_acc):>10.4f} {max(r_acc):>10.4f} {rb_max}")

    dqn_c    = str(dqn_conv)    if dqn_conv    else "미달"
    random_c = str(random_conv) if random_conv else "미달"
    rule_c   = str(rule_conv)   if rule_conv   else ("미달" if rule_results else "")
    print(f"{'수렴 라운드':20s} {dqn_c:>10s} {random_c:>10s} {rule_c:>12s}")

    dqn_he_v    = extract(dqn_m,    "avg_he_latency_norm")
    random_he_v = extract(random_m, "avg_he_latency_norm")
    rule_he_v   = extract(rule_m,   "avg_he_latency_norm") if rule_m else []
    if dqn_he_v and random_he_v:
        rb_he = f"{np.mean(rule_he_v):>12.4f}" if rule_he_v else f"{'N/A':>12s}"
        print(f"{'평균 HE Latency':20s} {np.mean(dqn_he_v):>10.4f} {np.mean(random_he_v):>10.4f} {rb_he}")

    dqn_rew_v    = extract(dqn_m,    "reward")
    random_rew_v = extract(random_m, "reward")
    rule_rew_v   = extract(rule_m,   "reward") if rule_m else []
    if dqn_rew_v and random_rew_v:
        rb_rew = f"{np.mean(rule_rew_v):>12.4f}" if rule_rew_v else f"{'N/A':>12s}"
        print(f"{'평균 Reward':20s} {np.mean(dqn_rew_v):>10.4f} {np.mean(random_rew_v):>10.4f} {rb_rew}")
    print("─" * 56)


def plot_epsilon_graph(dqn_results: dict, save_dir: str = "."):
    dqn_metrics = dqn_results.get("dqn_metrics", [])
    if not dqn_metrics:
        print("dqn_metrics 데이터가 없습니다.")
        return
    if "epsilon" not in dqn_metrics[0]:
        print("epsilon 데이터가 없습니다. dqn_strategy.py에 epsilon 저장 코드를 확인하세요.")
        return

    rounds  = [m["round"]   for m in dqn_metrics]
    epsilon = [m["epsilon"] for m in dqn_metrics]

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, epsilon, color=COLOR_DQN, linewidth=2)
    plt.title("DQN Epsilon per Round", fontsize=12, fontweight="bold")
    plt.xlabel("Round"); plt.ylabel("Epsilon")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path(save_dir) / "epsilon_per_round.png"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Epsilon 그래프 저장 완료: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn_path",    type=str, required=True)
    parser.add_argument("--random_path", type=str, required=True)
    parser.add_argument("--rule_path", "--rule_based_path", type=str, default=None,
                        dest="rule_path", help="rule-based pkl 경로 (없으면 2개 비교)")
    parser.add_argument("--save_dir",    type=str, default=".")
    parser.add_argument("--threshold",   type=float, default=0.90)
    args = parser.parse_args()

    dqn_results    = load(args.dqn_path)
    random_results = load(args.random_path)
    rule_results   = load(args.rule_path) if args.rule_path else None

    plot_comparison(
        dqn_results,
        random_results,
        rule_results=rule_results,
        save_dir=args.save_dir,
        conv_threshold=args.threshold,
    )
    plot_epsilon_graph(dqn_results, save_dir=args.save_dir)