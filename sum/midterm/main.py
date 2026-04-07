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

    # 1. config 출력
    print(OmegaConf.to_yaml(cfg))

    # 2. dataset 준비
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    # 3. client 생성
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)
    clients = [client_fn(str(i)) for i in range(cfg.num_clients)]

    # 4. edge 생성
    edges = []
    for i in range(cfg.num_edges):
        start = i * cfg.clients_per_edge
        end = (i + 1) * cfg.clients_per_edge
        edge_clients = clients[start:end]
        edges.append(EdgeServer(edge_clients))

    # 5. Main Server 생성 (🔥 testloader 전달 중요)
    server = MainServer(edges, cfg.num_classes, testloader)

    # 6. fit config 설정
    fit_config = dict(cfg.config_fit)
    fit_config["clients_per_edge_per_round"] = cfg.clients_per_edge_per_round

    # 7. 학습 실행
    history = server.run(cfg.num_rounds, fit_config)

    # 8. 결과 저장
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {
        "history": history
    }

    with open(str(results_path), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()