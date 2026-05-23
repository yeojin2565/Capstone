# 실험 보고서

# Experiment Setup

## 1. Architecture
- 1 Cloud Server 100 clients
- 클라이언트에서 2번의 epoch 수행 후 서버로 전송, FedAvg 후 글로벌 모델 업데이트(Flower 프레임워크)

## 2. Dataset
- CIFAR-10
- non-IID: Dirichlet(alpha=0.5) 분포로 class 불균형

## 3. Clients Design
|그룹|수량|HE latency 분포|dropout 확률|특성|
|---|---|---|---|---|
|Excellent|5개|$N(0.02, 0.005)$|0.02|극단적으로 좋음|
|Fast|30개|$N(0.1, 0.02)$|0.05|빠름|
|Medium|30개|$N(0.5, 0.08)$|0.10|보통|
|Slow|25개|$N(1.5, 0.25)$|0.20|느림|
|Extreme|10개|$N(5.0, 0.50)$|0.40|극단적으로 나쁨|

```python
"""he_simulator.py"""
GROUP_CONFIG = {
    "excellent": {"mean": 0.02, "std": 0.005, "dropout": 0.02},
    "fast":      {"mean": 0.10, "std": 0.020, "dropout": 0.05},
    "medium":    {"mean": 0.50, "std": 0.080, "dropout": 0.10},
    "slow":      {"mean": 1.50, "std": 0.250, "dropout": 0.20},
    "extreme":   {"mean": 5.00, "std": 0.500, "dropout": 0.40},
}
```

### (a) Metric
```python
"""client.py"""
metrics = {
    "loss":          float(loss),            "accuracy":      float(accuracy),
    "train_latency": float(train_latency),
    "he_latency":    float(he_latency),
    "data_size":     len(self.train_indices),
    "dropped":       0,
    "cid":           self.cid,
}
```
### (b) Dropout
- **Dropout**: DQN이 선택했음에도 클라이언트가 글로벌 업데이트에 기여하지 못하는 상태

```python
"""he_simulator.py"""
def simulate_dropout(cid: int) -> bool:
    """
    그룹별 확률로 dropout 시뮬레이션
    True: 해당 라운드 탈락
    """
    group = get_group(cid)
    prob  = GROUP_CONFIG[group]["dropout"]
    return np.random.random() < prob
```

### (c) Client ID Mapping
```python
"""he_simulator.py"""
def get_group(cid: int) -> str:
    if cid < 5:    return "excellent"
    elif cid < 35: return "fast"
    elif cid < 65: return "medium"
    elif cid < 90: return "slow"
    else:          return "extreme"
```

### (d) Transmission Noise
- 전송 중 노이즈로 latency 발생
- 분포: $N(0, 0.01)$
```python
"""he_simulator.py"""
TRANSMISSION_NOISE_STD = 0.01
HE_LATENCY_MAX = 6.0
```

### (e) HE latency
- 매 라운드마다 HE latency 계산
- latency = HE latency + Transmission Noise
- max: 6.0
- [0, 1] 정규화: `float(np.clip(latency / HE_LATENCY_MAX, 0.0, 1.0))`
```python
"""he_simulator.py"""
TRANSMISSION_NOISE_STD = 0.01
HE_LATENCY_MAX = 6.0

def simulate_he_latency(base_latency: float) -> float:
    noise   = np.random.normal(0, TRANSMISSION_NOISE_STD)
    latency = base_latency + noise
    return float(np.clip(latency, 0.005, HE_LATENCY_MAX))
```

## 4. DQN
### (a) State
- $C_i = [h_i, d_i]$ 
- $h_i = \text{HE latency of i-th client}, d_i = \text{data size of i-th client}$

### (b) Action
- DQN output: 100개 clients 각각에 대한 score
- $Q ≈ \frac{1}{k}\sum_{i∈a} \text{score}(i)$
```python
"""dqn.py"""
curr_scores = self.model(states_t)                        # [B, n_clients]
curr_q      = (curr_scores * actions_t).sum(1) / self.k_select
```

### (c) Reward
- $R = w_1 * \Delta\text{accuracy} + w_2 * \text{avg quality bounus} - w_3 * \text{avg HE latency} - w_4 * \text{dropout rate} + \text{fast bouns} - \text{slow penalty}$
- $\text{quality bonus} = \text{data size}*(1 - \text{HE latency})$
- $\text{dropout rate} = \frac{\text{dropout count}}{k}, k = \text{num of selected clients}$
- $\text{fast bonus} = \alpha*\frac{\text{num of fast clients}}{k}, \alpha = \text{HE bonus value}=0.25$
- $\text{slow penalty} = \beta*\frac{num of slow clients}{k}, \beta=\text{HE slow penalty value}=0.20$

# 실험 결과

## FL Hyperparameter
|parameter|value|
|---|---|
|rounds|200|
|num of clients|100|
|num of clients per round|10|
|batch size|32|
|learning rate|0.01|
|momentum|0.9|
|local epochs|2|

## DQN Hyperparameter
|parameter|value|
|---|---|
|discount factor $\gamma$|0.95|
|$\epsilon$ start|1.0|
|$\epsilon$ decay|0.95|
|$\epsilon$ min|0.05|
|batch size|32|
|memory size|5000|
|target update|5 step|