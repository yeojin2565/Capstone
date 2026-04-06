from crypto import decrypt_weights
import random


class EdgeServer:

    def __init__(self, clients):
        self.clients = clients
        self.last_selected_clients = []

    # 암호 상태에서 weight 합산
    def aggregate_encrypted(self, results):

        agg = None

        for weights, _ in results:
            if agg is None:
                agg = weights
            else:
                for i in range(len(agg)):
                    for j in range(len(agg[i])):
                        agg[i][j] += weights[i][j]

        return agg

    def fit(self, global_weights, config):

        results = []

        k = config["clients_per_edge_per_round"]

        selected_clients = random.sample(self.clients, k)
        self.last_selected_clients = selected_clients

        for client in selected_clients:
            weights, data_size, _ = client.fit(global_weights, config)
            results.append((weights, data_size))

        # 1. 암호 상태에서 weight 합산
        enc_sum = self.aggregate_encrypted(results)

        # 2. 복호화
        shapes = [w.shape for w in global_weights]
        dec_sum = decrypt_weights(enc_sum, shapes)

        # 3. 평균 계산
        total_clients = len(results)
        new_weights = [w / total_clients for w in dec_sum]

        return new_weights, total_clients

    def get_states(self):

        states = []

        for client in self.last_selected_clients:
            state = client.get_state()
            if state is not None:
                states.append(state)

        return states