# DQN-based Client Selection in Federated Learning

Undergraduate Capstone Project — 2026

## Overview

This project proposes a **DQN-based client selection strategy for Federated Learning (FL)** that incorporates **HE (Homomorphic Encryption) latency** as a key state feature. The DQN agent learns to preferentially select clients with lower encryption overhead, reducing the overall communication cost while maintaining model accuracy.

Clients are heterogeneous in terms of HE computation capability, modeled via Gaussian distributions across five performance groups. A random selection baseline is provided for comparison.

---

## Key Features

- **DQN client selection**: learns to avoid high-latency clients over rounds
- **Heterogeneous client simulation**: 5 groups (Excellent / Fast / Medium / Slow / Extreme) with Gaussian HE latency distributions
- **Transmission noise**: random noise added to HE latency each round to simulate real network conditions
- **Dropout simulation**: higher-latency clients have higher dropout probability
- **Non-IID data**: CIFAR-10 distributed via Dirichlet(α=0.5)
- **Random selection baseline**: for fair comparison

---

## Project Structure

```
├── conf/
│   └── base.yaml            # Hydra config (rounds, clients, lr, etc.)
│
├── he_simulator.py          # HE latency simulation (Gaussian per group)
├── model.py                 # CIFAR-10 CNN (3-channel input)
├── dataset.py               # CIFAR-10 Non-IID data split (Dirichlet)
├── client.py                # Flower client with HE latency simulation
├── server.py                # Fit config + global model evaluation
│
├── dqn.py                   # DQN agent (QNetwork, replay memory)
├── dqn_strategy.py          # FedAvg + DQN client selection
├── random_strategy.py       # FedAvg + random selection (baseline)
│
├── train_dqn.py             # Run DQN experiment
├── train_random.py          # Run random baseline experiment
├── compare_results.py       # Plot DQN vs Random comparison graphs
│
├── Experiment_setup.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Client Groups

| Group     | Count | HE Latency Distribution | Dropout |
|-----------|-------|--------------------------|---------|
| Excellent | 3     | N(0.02, 0.005)           | 2%      |
| Fast      | 10    | N(0.10, 0.020)           | 5%      |
| Medium    | 10    | N(0.50, 0.080)           | 10%     |
| Slow      | 10    | N(1.50, 0.250)           | 20%     |
| Extreme   | 3     | N(5.00, 0.500)           | 40%     |

---

## State / Action / Reward

**State** (180-dim): `[loss, accuracy, train_latency, he_latency, data_size]` × 36 clients, normalized

**Action**: Top-K clients by Q-value (K=10)

**Reward**:
```
R = -avg_he_latency_norm + α * avg_accuracy - β * dropout_count
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run

```bash
# DQN experiment
python train_dqn.py

# Random baseline
python train_random.py

# Comparison plot
python compare_results.py \
    --dqn_path    outputs/YYYY-MM-DD/HH-MM-SS/results.pkl \
    --random_path outputs/YYYY-MM-DD/HH-MM-SS/results_random.pkl
```

---

## Results

After training, `compare_results.py` generates a 2×2 comparison plot:

| Graph | Description |
|---|---|
| Global Loss | Loss curve per round |
| Global Accuracy | Accuracy + convergence round |
| Avg HE Latency | Core contribution — DQN should be lower |
| Reward per Round | DQN reward vs random |

---

## Requirements

- Python 3.10+
- PyTorch
- Flower (flwr)
- Ray
- Hydra-core
- NumPy, Matplotlib