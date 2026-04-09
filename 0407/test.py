# test.py

import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from DQN import DQNAgent
from DQN_strategy import FedAvgWithDQN, N_CLIENTS, STATE_SIZE
from he_utils import create_he_context, get_public_context


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## 1. config 출력
    print(OmegaConf.to_yaml(cfg))
    
    # ── HE 컨텍스트 생성 (서버 1회) ─────────────────
    print("── HE 컨텍스트 생성 중... (약 5~10초 소요) ──")
    he_context_full   = create_he_context()                  # 비밀키 포함 (서버용)
    he_context_public = get_public_context(he_context_full)  # 공개키만 (클라이언트용)
    he_context_public_bytes = he_context_public.serialize()
    print("── HE 컨텍스트 생성 완료 ──")

    ## 2. 데이터 준비
    trainloaders, validationloaders, testloaders = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )
    

    ## 3. 클라이언트 정의
    client_fn = generate_client_fn(
        trainloaders, validationloaders, cfg.num_classes,
        he_context_bytes=he_context_public_bytes,
        )

    ## 4. DQN 에이전트 초기화
    agent = DQNAgent(
        state_size=STATE_SIZE,
        n_clients=N_CLIENTS,
        k_select=cfg.num_clients_per_round_fit,
    )

    ## 5. DQN 커스텀 전략 정의 (비밀키 컨텍스트 전달)
    strategy = FedAvgWithDQN(
        dqn_agent=agent,
        he_context=he_context_full,
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(
            cfg.num_classes, testloaders, he_context_full
            ),
    )

    ## 6. 시뮬레이션 실행
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        client_resources={"num_cpus": 0.5, "num_gpus": 0},
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 2, 
                       "num_gpus": 0, 
                       "include_dashboard": False,
                       "ignore_reinit_error": True
                       },
    )

    ## 7. 결과 저장
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {
        "history":        history,
        "dqn_metrics":    strategy.history_metrics, 
    }

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    ## 8. DQN 모델 저장
    agent.save(str(Path(save_path) / "dqn_model.pth"))

    ## 9. 간단한 결과 출력
    print("\n── 학습 완료 ──")
    for m in strategy.history_metrics:
        print(
            f"Round {m['round']:02d} | "
            f"accuracy={m['accuracy']:.4f} | "
            f"he_latency={m['avg_he_latency_norm']:.4f} | "
            f"reward={m['reward']:.4f} | "
        )


if __name__ == "__main__":
    main()