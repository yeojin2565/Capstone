# Experiment Setup

## 1. Architecture
- 1 Cloud Server 36 clients
- 클라이언트에서 2번의 epoch 수행 후 서버로 전송, FedAvg 후 글로벌 모델 업데이트(Flower 프레임워크)

## 2. Clients Design
|그룹|수량|HE latency 분포|특성|
|---|---|---|---|
|Excellent|3개|$N(0.02, 0.005)$|극단적으로 좋음|
|Fast|10개|$N(0.1, 0.02)$|빠름|
|Medium|10개|$N(0.5, 0.08)$|보통|
|Slow|10개|$N(1.5, 0.25)$|느림|
|Extreme|3개|$N(5.0, 0.50)$|극단적으로 나쁨|

### (a) Features
```python
# 클라이언트별 고정 특성
static_features = { "he_latency_base": float }

# 매 라운드 변하는 동적 특성
dynamic_features = {
    "loss":            float,  # 로컬 학습 후 val loss
    "accuracy":        float,  # 로컬 학습 후 val accuracy
    "train_latency":   float,  # 로컬 학습 소요 시간
    "he_latency":      float,  # he_latency_base + 전송 노이즈
                               # noise ~ N(0, 0.01)
    "data_size":       int,    # 보유 데이터 수
}
```
### (b) Dropout
- **Dropout**: DQN이 선택했음에도 클라이언트가 글로벌 업데이트에 기여하지 못하는 상태
```python
HE_DROPOUT = {
    "excellent": 0.02,   # 2%  (안정적인 고성능 기기)
    "fast":      0.05,   # 5%
    "medium":    0.10,   # 10%
    "slow":      0.20,   # 20%
    "extreme":   0.40,   # 40%
}
```

### (c) Client ID Mapping Function
```python
def get_group(cid: int) -> str:
    if cid < 3:           return "excellent"  # 0, 1, 2
    elif cid < 13:        return "fast"       # 3~12
    elif cid < 23:        return "medium"     # 13~22
    elif cid < 33:        return "slow"       # 23~32
    else:                 return "extreme"    # 33, 34, 35
```

## 3. State/Action/Reward
### (a) State
- 36 clients × 5 features = 180
```python
# 36개 클라이언트 × 5개 feature = 180차원
state = flatten([
    [loss_i, accuracy_i, train_latency_i, he_latency_i, data_size_i]
    for i in range(36)
])  # shape: (180,)
```

### (b) Action
- 36 clients 중 k개 선택 (k = 10)
- Q값 상위 k개 인덱스 반환
```python
action = top_k(Q_values, k=10) # shape: (10,)
```

### (c) Reward
$R = -\text{average HE latency} +\alpha * \text{average accuracy} - \beta * dropout$

# 4. 요약
|항목|설정값|
|---|---|
|Num of clients|36|
|선택되는 클라이언트 수(k)|10|
|dataset|CIFAR-10(Non-IID, Dirichlet α=0.5)|
|Local model|CNN(3채널 입력)|
|HE latency|그룹별 가우시안 + 전송 노이즈|
|State 차원|180(36 × 5)|
|Action|Q값 상위 10개 인덱스|
|Reward|-HE latency + α*accuracy - β\*dropout|
|RL algorithm|DQN(epsilon-greedy)|
