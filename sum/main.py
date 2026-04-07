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

    edges = []
    for i in range(cfg.num_edges):
        start = i * cfg.clients_per_edge
        end = (i + 1) * cfg.clients_per_edge
        edge_clients = clients[start:end]

        n_clients = len(edge_clients)
        state_size = n_clients * state_per_client

        edges.append(
            EdgeServer(
                edge_clients,
                state_size,
                n_clients,
                cfg.clients_per_edge_per_round
            )
        )

    server = MainServer(edges, cfg.num_classes, testloader)

    fit_config = dict(cfg.config_fit)
    fit_config["clients_per_edge_per_round"] = cfg.clients_per_edge_per_round
    fit_config["warmup_rounds"] = cfg.warmup_rounds

    history = server.run(cfg.num_rounds, fit_config)

    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {
        "history": history
    }

    with open(str(results_path), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()