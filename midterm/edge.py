from crypto import decrypt_weights
import random


class EdgeServer:

    def __init__(self, clients):
        self.clients = clients
        self.last_selected_clients = []

    # 암호 상태에서는 덧셈만 수행
    def aggregate_encrypted(self, results):

        agg = None

        for delta, _ in results:
            if agg is None:
                agg = delta
            else:
                for i in range(len(agg)):
                    agg[i] += delta[i]

        return agg

    def fit(self, global_weights, config):

        results = []

        k = config["clients_per_edge_per_round"]

        # client 선택
        selected_clients = random.sample(self.clients, k)
        self.last_selected_clients = selected_clients

        # local 학습 수행
        for client in selected_clients:
            delta, data_size, _ = client.fit(global_weights, config)
            results.append((delta, data_size))

        # 1. 암호 상태에서 delta 합산
        enc_sum = self.aggregate_encrypted(results)

        # 2. 복호화
        shapes = [w.shape for w in global_weights]
        dec_sum = decrypt_weights(enc_sum, shapes)

        # 3. 평균 delta 계산
        total_clients = len(results)
        avg_delta = [d / total_clients for d in dec_sum]

        # 4. global weight 업데이트
        new_weights = [
            w + d
            for w, d in zip(global_weights, avg_delta)
        ]

        return new_weights, total_clients

    # 선택된 client들의 state 반환
    def get_states(self):

        states = []

        for client in self.last_selected_clients:
            state = client.get_state()
            if state is not None:
                states.append(state)

        return states