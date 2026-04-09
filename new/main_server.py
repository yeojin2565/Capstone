# main_server.py
import os
import random
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")   # plt.show() 없이 파일 저장 (GUI 없는 환경 대응)
                        # ※ 반드시 pyplot import 전에 호출해야 함
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from crypto import decrypt_weights

from model import Net, test
import torch

# ── 색상 ──────────────────────────────────────────────
COLOR_DQN    = "#1D9E75"   # 초록  (제안 방법)
COLOR_RANDOM = "#D85A30"   # 주황  (baseline)
COLOR_SHADE  = 0.10        # fill_between 투명도


# ── 이동 평균 추세선 ───────────────────────────────────
def add_moving_average(ax, x, y, color, window=5):
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
    # list도 안전하게 슬라이싱하기 위해 numpy 배열로 변환
    x_arr = np.array(x)
    x_ma  = x_arr[window - 1:]
    ax.plot(
        x_ma, ma,
        color=color, linewidth=2.2,
        linestyle="-", alpha=0.85,
        label="_nolegend_",
    )


# ── 수렴 라운드 ────────────────────────────────────────
def convergence_round(accuracies, threshold):
    # type: (list, float) -> object
    for i, a in enumerate(accuracies):
        if a >= threshold:
            return i + 1
    return None


class MainServer:

    def __init__(self, edges, num_classes, testloader):
        self.edges      = edges
        self.testloader = testloader
        self.device     = torch.device("cpu")

        # global model 초기화
        self.model          = Net(num_classes).to(self.device)
        self.global_weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        self.history = []

    # Edge 결과를 FedAvg로 aggregation
    def aggregate(self, edge_results):
        total_clients = 0
        agg_weights   = None

        for weights, num_clients in edge_results:
            if agg_weights is None:
                agg_weights = [w * num_clients for w in weights]
            else:
                for i in range(len(agg_weights)):
                    agg_weights[i] += weights[i] * num_clients
            total_clients += num_clients

        avg_weights = [w / total_clients for w in agg_weights]
        return avg_weights

    # 한 라운드 학습
    def train_round(self, fit_config):
        edge_results = []
        for edge in self.edges:
            weights, num_clients = edge.fit(self.global_weights, fit_config)
            edge_results.append((weights, num_clients))
        self.global_weights = self.aggregate(edge_results)

    # 평가
    def evaluate(self):
        state_dict = dict(zip(self.model.state_dict().keys(), self.global_weights))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

        loss, acc = test(self.model, self.testloader, self.device)
        print(f"Global Loss: {loss:.4f}, Global Accuracy: {acc:.4f}")
        return {"loss": loss, "accuracy": acc}

    # 모든 edge에서 state 수집
    def get_states(self):
        all_states = []
        for edge in self.edges:
            edge_states = edge.get_states()
            all_states.extend(edge_states)
        return all_states

    # 모든 edge에서 history_metrics 수집 → round 기준으로 평균 집계
    def get_dqn_metrics(self):
        # edge가 여러 개면 같은 round에 데이터포인트가 여러 개 생겨 그래프가 엉킴
        # → round별로 묶어서 평균을 내고 1개의 데이터포인트로 만든다
        round_buckets = defaultdict(list)
        for edge in self.edges:
            for m in edge.history_metrics:
                round_buckets[m["round"]].append(m)

        aggregated = []
        for rnd in sorted(round_buckets.keys()):
            bucket = round_buckets[rnd]
            # dqn_loss는 None일 수 있으므로 None 제외 후 평균
            valid_losses = [m["dqn_loss"] for m in bucket if m["dqn_loss"] is not None]
            aggregated.append({
                "round":               rnd,
                "accuracy":            float(np.mean([m["accuracy"]            for m in bucket])),
                "avg_he_latency_norm": float(np.mean([m["avg_he_latency_norm"] for m in bucket])),
                "reward":              float(np.mean([m["reward"]              for m in bucket])),
                "dqn_loss":            float(np.mean(valid_losses)) if valid_losses else None,
                "epsilon":             float(np.mean([m["epsilon"]             for m in bucket])),
            })
        return aggregated

    # 전체 학습 루프
    def run(self, num_rounds, fit_config):
        for rnd in range(num_rounds):
            print(f"\n===== Round {rnd+1} =====")
            self.train_round(fit_config)

            # 현재 라운드 state 수집
            states = self.get_states()
            print("Collected States:", states)

            result = self.evaluate()
            self.history.append({
                "round":    rnd + 1,
                "loss":     result["loss"],
                "accuracy": result["accuracy"]
            })

        return self.history

    # ── 랜덤 baseline 실행 (DQN 비교용) ──────────────────
    def run_random_baseline(self, num_rounds, fit_config):
        """
        클라이언트를 완전 랜덤으로 K개 선택하는 baseline 실험.
        DQN 전략과 동일한 history 구조를 유지해서 비교 그래프를 그릴 수 있도록 한다.
        """
        k = fit_config["clients_per_edge_per_round"]

        random_history = []
        random_metrics = []  # avg_he_latency_norm, reward 기록

        for rnd in range(num_rounds):
            print(f"\n===== [Random Baseline] Round {rnd+1} =====")

            edge_results = []
            round_he     = []

            for edge in self.edges:
                # ── 랜덤 클라이언트 선택 ─────────────────
                selected_ids     = random.sample(range(len(edge.clients)), k)
                selected_clients = [edge.clients[i] for i in selected_ids]

                print(f"[Random] Selected clients: {selected_ids}")

                results = []
                for client in selected_clients:
                    weights, data_size, _ = client.fit(self.global_weights, fit_config)
                    results.append((weights, data_size))

                # 암호화된 가중치 합산 후 복호화
                enc_sum = edge.aggregate_encrypted(results)
                shapes  = [w.shape for w in self.global_weights]
                dec_sum = decrypt_weights(enc_sum, shapes)

                total_clients = len(results)
                new_weights   = [w / total_clients for w in dec_sum]
                edge_results.append((new_weights, total_clients))

                # he_latency 수집
                for client in selected_clients:
                    s = client.get_state()
                    if s is not None:
                        he_norm = float(np.clip(s["he_latency"] / 1.0, 0.0, 1.0))
                        round_he.append(he_norm)

            self.global_weights = self.aggregate(edge_results)

            # ── 평가 ─────────────────────────────────
            result = self.evaluate()

            avg_he  = float(np.mean(round_he))  if round_he  else 0.5
            reward  = -avg_he                   # DQN과 동일한 reward 공식 (공정한 비교)

            random_history.append({
                "round":    rnd + 1,
                "loss":     result["loss"],
                "accuracy": result["accuracy"]
            })
            random_metrics.append({
                "round":               rnd + 1,
                "accuracy":            result["accuracy"],
                "avg_he_latency_norm": avg_he,
                "reward":              reward,
            })

        return random_history, random_metrics

    # ── 비교 그래프 저장 ───────────────────────────────────
    def plot_comparison(
        self,
        dqn_history,
        dqn_metrics,
        random_history,
        random_metrics,
        save_dir=".",
        conv_threshold=0.90,
    ):
        """
        DQN 제안 방법 vs Random baseline 비교 그래프 (2행 3열, 총 5개 + 요약)
          [0,0] Global Loss     [0,1] Global Accuracy   [0,2] HE Latency
          [1,0] Reward          [1,1] Epsilon Decay      [1,2] 수치 요약
        plt.show() 대신 파일로 저장 (os.makedirs 사용)
        """
        # ── 데이터 파싱 ──────────────────────────────
        d_rounds = [m["round"]    for m in dqn_history]
        d_loss   = [m["loss"]     for m in dqn_history]
        d_acc    = [m["accuracy"] for m in dqn_history]

        r_rounds = [m["round"]    for m in random_history]
        r_loss   = [m["loss"]     for m in random_history]
        r_acc    = [m["accuracy"] for m in random_history]

        dqn_rounds = [m["round"]               for m in dqn_metrics]
        dqn_he     = [m["avg_he_latency_norm"]  for m in dqn_metrics]
        dqn_rew    = [m["reward"]               for m in dqn_metrics]
        dqn_eps    = [m["epsilon"]              for m in dqn_metrics]

        random_rounds = [m["round"]              for m in random_metrics]
        random_he     = [m["avg_he_latency_norm"] for m in random_metrics]
        random_rew    = [m["reward"]              for m in random_metrics]

        # ── 수렴 라운드 ──────────────────────────────
        dqn_conv    = convergence_round(d_acc, conv_threshold)
        random_conv = convergence_round(r_acc, conv_threshold)
        dqn_c_str    = str(dqn_conv)    if dqn_conv    else "N/A"
        random_c_str = str(random_conv) if random_conv else "N/A"

        # ── 레이아웃: 2행 3열 ──────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "DQN Client Selection vs Random Baseline",
            fontsize=14, fontweight="bold", y=0.98
        )

        def legend_labels():
            return [
                Line2D([0], [0], color=COLOR_DQN,   linewidth=2, label="DQN (proposed)"),
                Line2D([0], [0], color=COLOR_RANDOM, linewidth=2, label="Random (baseline)"),
            ]

        # ── 1. Global Loss ─────────────────────────────────
        ax = axes[0][0]
        if d_loss:
            ax.plot(d_rounds, d_loss, color=COLOR_DQN,    linewidth=1.5, alpha=0.35)
            ax.fill_between(d_rounds, d_loss, alpha=COLOR_SHADE, color=COLOR_DQN)
            add_moving_average(ax, d_rounds, d_loss, COLOR_DQN)
        if r_loss:
            ax.plot(r_rounds, r_loss, color=COLOR_RANDOM, linewidth=1.5, alpha=0.35)
            ax.fill_between(r_rounds, r_loss, alpha=COLOR_SHADE, color=COLOR_RANDOM)
            add_moving_average(ax, r_rounds, r_loss, COLOR_RANDOM)
        ax.set_title("Global Loss", fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.legend(handles=legend_labels(), fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        # ── 2. Global Accuracy ─────────────────────────────
        ax = axes[0][1]
        if d_acc:
            ax.plot(d_rounds, d_acc, color=COLOR_DQN,    linewidth=1.5, alpha=0.35)
            ax.fill_between(d_rounds, d_acc, alpha=COLOR_SHADE, color=COLOR_DQN)
            add_moving_average(ax, d_rounds, d_acc, COLOR_DQN)
        if r_acc:
            ax.plot(r_rounds, r_acc, color=COLOR_RANDOM,  linewidth=1.5, alpha=0.35)
            ax.fill_between(r_rounds, r_acc, alpha=COLOR_SHADE, color=COLOR_RANDOM)
            add_moving_average(ax, r_rounds, r_acc, COLOR_RANDOM)
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
        ax = axes[0][2]
        if dqn_he:
            ax.plot(dqn_rounds, dqn_he, color=COLOR_DQN,    linewidth=1.5, alpha=0.35)
            ax.fill_between(dqn_rounds, dqn_he, alpha=COLOR_SHADE, color=COLOR_DQN)
            add_moving_average(ax, dqn_rounds, dqn_he, COLOR_DQN)
        if random_he:
            ax.plot(random_rounds, random_he, color=COLOR_RANDOM, linewidth=1.5, alpha=0.35)
            ax.fill_between(random_rounds, random_he, alpha=COLOR_SHADE, color=COLOR_RANDOM)
            add_moving_average(ax, random_rounds, random_he, COLOR_RANDOM)
        ax.set_title("Avg HE Latency (normalized)", fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("HE Latency (norm)")
        ax.set_ylim(0, 1.05)
        ax.legend(handles=legend_labels(), fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        # ── 4. Reward ──────────────────────────────────────
        ax = axes[1][0]
        if dqn_rew:
            ax.plot(dqn_rounds, dqn_rew, color=COLOR_DQN,    linewidth=1.5, alpha=0.35)
            ax.fill_between(dqn_rounds, dqn_rew, alpha=COLOR_SHADE, color=COLOR_DQN)
            add_moving_average(ax, dqn_rounds, dqn_rew, COLOR_DQN)
        if random_rew:
            ax.plot(random_rounds, random_rew, color=COLOR_RANDOM, linewidth=1.5, alpha=0.35)
            ax.fill_between(random_rounds, random_rew, alpha=COLOR_SHADE, color=COLOR_RANDOM)
            add_moving_average(ax, random_rounds, random_rew, COLOR_RANDOM)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("Reward per Round", fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Reward")
        ax.legend(handles=legend_labels(), fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        # ── 5. Epsilon Decay (DQN 전용) ────────────────────
        ax = axes[1][1]
        if dqn_eps:
            ax.plot(dqn_rounds, dqn_eps, color=COLOR_DQN, linewidth=2.0)
            ax.fill_between(dqn_rounds, dqn_eps, alpha=COLOR_SHADE, color=COLOR_DQN)
        ax.set_title("Epsilon Decay (DQN)", fontsize=12)
        ax.set_xlabel("Round")
        ax.set_ylabel("Epsilon")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        # ── 6. 수치 요약 텍스트 ────────────────────────────
        ax = axes[1][2]
        ax.axis("off")  # 축 숨김, 텍스트만 표시
        lines = ["Summary", "─" * 32]
        lines.append(f"{'':22s} {'DQN':>6s} {'Rand':>6s}")
        lines.append("─" * 32)
        if d_acc and r_acc:
            lines.append(f"{'Final Accuracy':22s} {d_acc[-1]:>6.4f} {r_acc[-1]:>6.4f}")
            lines.append(f"{'Best Accuracy':22s} {max(d_acc):>6.4f} {max(r_acc):>6.4f}")
        lines.append(f"{'Convergence Round':22s} {dqn_c_str:>6s} {random_c_str:>6s}")
        if dqn_he and random_he:
            lines.append(f"{'Avg HE Latency':22s} {np.mean(dqn_he):>6.4f} {np.mean(random_he):>6.4f}")
        if dqn_rew and random_rew:
            lines.append(f"{'Avg Reward':22s} {np.mean(dqn_rew):>6.4f} {np.mean(random_rew):>6.4f}")
        if dqn_eps:
            lines.append(f"{'Final Epsilon':22s} {dqn_eps[-1]:>6.4f} {'  -':>6s}")
        lines.append("─" * 32)
        ax.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8),
        )

        plt.tight_layout()

        # ── plt.show() 대신 파일로 저장 (os.makedirs 사용) ──
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "comparison.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"비교 그래프 저장 완료: {out}")

        # ── 수치 요약 콘솔 출력 ────────────────────────────
        print("\n──────────────────────────────────────")
        print(f"{'':20s} {'DQN':>10s} {'Random':>10s}")
        print("──────────────────────────────────────")
        if d_acc and r_acc:
            print(f"{'최종 Accuracy':20s} {d_acc[-1]:>10.4f} {r_acc[-1]:>10.4f}")
            print(f"{'최고 Accuracy':20s} {max(d_acc):>10.4f} {max(r_acc):>10.4f}")
        print(f"{'수렴 라운드':20s} {dqn_c_str:>10s} {random_c_str:>10s}")
        if dqn_he and random_he:
            print(f"{'평균 HE Latency':20s} {np.mean(dqn_he):>10.4f} {np.mean(random_he):>10.4f}")
        if dqn_rew and random_rew:
            print(f"{'평균 Reward':20s} {np.mean(dqn_rew):>10.4f} {np.mean(random_rew):>10.4f}")
        if dqn_eps:
            print(f"{'최종 Epsilon':20s} {dqn_eps[-1]:>10.4f}")
        print("──────────────────────────────────────")
