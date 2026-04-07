from model import Net, test
import torch


class MainServer:

    def __init__(self, edges, num_classes, testloader):
        self.edges = edges
        self.testloader = testloader
        self.device = torch.device("cpu")

        # global model 초기화
        self.model = Net(num_classes).to(self.device)
        self.global_weights = [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

        self.history = []

    # Edge 결과를 FedAvg로 aggregation
    def aggregate(self, edge_results):

        total_clients = 0
        agg_weights = None

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

        # global aggregation
        self.global_weights = self.aggregate(edge_results)

    # 평가
    def evaluate(self):

        state_dict = dict(
            zip(self.model.state_dict().keys(), self.global_weights)
        )

        self.model.load_state_dict({
            k: torch.tensor(v) for k, v in state_dict.items()
        })

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

    # 전체 학습 루프
    def run(self, num_rounds, fit_config):

        for rnd in range(num_rounds):
            print(f"\n===== Round {rnd+1} =====")

            self.train_round(fit_config)

            # 현재 라운드 state 수집 (RL 입력용)
            states = self.get_states()
            print("Collected States:", states)

            result = self.evaluate()

            self.history.append({
                "round": rnd + 1,
                "loss": result["loss"],
                "accuracy": result["accuracy"]
            })

        return self.history