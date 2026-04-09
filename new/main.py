import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from dataset import prepare_dataset
from client import generate_client_fn
from edge import EdgeServer
from main_server import MainServer


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    client_fn = generate_client_fn(
        trainloaders,
        validationloaders,
        cfg.num_classes
    )
    clients = [client_fn(str(i)) for i in range(cfg.num_clients)]

    # ----- state_size 정의 -----
    # client state: [loss, acc, train_latency, he_latency, data_size]
    state_per_client = 5

    fit_config = dict(cfg.config_fit)
    fit_config["clients_per_edge_per_round"] = cfg.clients_per_edge_per_round
    fit_config["warmup_rounds"]              = cfg.warmup_rounds

    save_path = HydraConfig.get().runtime.output_dir

    # ══════════════════════════════════════════
    ## 1. DQN 실험
    # ══════════════════════════════════════════
    print("\n── DQN 실험 ──")

    edges_dqn = []
    for i in range(cfg.num_edges):
        start        = i * cfg.clients_per_edge
        end          = (i + 1) * cfg.clients_per_edge
        edge_clients = clients[start:end]

        n_clients  = len(edge_clients)
        state_size = n_clients * state_per_client

        edges_dqn.append(
            EdgeServer(
                edge_clients,
                state_size,
                n_clients,
                cfg.clients_per_edge_per_round
            )
        )

    server_dqn  = MainServer(edges_dqn, cfg.num_classes, testloader)
    dqn_history = server_dqn.run(cfg.num_rounds, fit_config)
    dqn_metrics = server_dqn.get_dqn_metrics()

    # ── DQN 결과 저장 ──────────────────────────
    results_dqn_path = Path(save_path) / "results.pkl"
    with open(str(results_dqn_path), "wb") as f:
        pickle.dump(
            {"history": dqn_history, "dqn_metrics": dqn_metrics},
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # ══════════════════════════════════════════
    ## 2. Random Baseline 실험
    # ══════════════════════════════════════════
    print("\n── Random Baseline 실험 ──")

    # 동일 클라이언트 재사용 (공정한 비교)
    # ※ 주의: DQN 실험에서 이미 학습된 클라이언트 상태가 남아 있음.
    #   완전히 독립적인 비교를 원하면 trainloaders/valloaders로 clients를 재생성할 것.
    #   현재는 동일 데이터 분포 조건 유지 목적으로 재사용함.
    edges_rand = []
    for i in range(cfg.num_edges):
        start        = i * cfg.clients_per_edge
        end          = (i + 1) * cfg.clients_per_edge
        edge_clients = clients[start:end]

        n_clients  = len(edge_clients)
        state_size = n_clients * state_per_client

        edges_rand.append(
            EdgeServer(
                edge_clients,
                state_size,
                n_clients,
                cfg.clients_per_edge_per_round
            )
        )

    server_rand = MainServer(edges_rand, cfg.num_classes, testloader)

    # warmup_rounds=num_rounds → 항상 랜덤 선택 (DQN 학습 없음)
    rand_config = dict(fit_config)
    rand_config["warmup_rounds"] = cfg.num_rounds

    random_history, random_metrics = server_rand.run_random_baseline(
        cfg.num_rounds, rand_config
    )

    # ── Random 결과 저장 ────────────────────────
    results_rand_path = Path(save_path) / "results_random.pkl"
    with open(str(results_rand_path), "wb") as f:
        pickle.dump(
            {"history": random_history, "dqn_metrics": random_metrics, "method": "random"},
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # ══════════════════════════════════════════
    ## 3. 비교 그래프 저장
    # ══════════════════════════════════════════
    server_dqn.plot_comparison(
        dqn_history    = dqn_history,
        dqn_metrics    = dqn_metrics,
        random_history = random_history,
        random_metrics = random_metrics,
        save_dir       = save_path,
        conv_threshold = getattr(cfg, "conv_threshold", 0.90),
    )

    ## 4. 간단한 결과 출력
    print("\n── 학습 완료 ──")
    for m in dqn_metrics:
        print(
            "Round {:02d} | accuracy={:.4f} | reward={:.4f} | epsilon={:.3f}".format(
                m["round"], m["accuracy"], m["reward"], m["epsilon"]
            )
        )


if __name__ == "__main__":
    main()
