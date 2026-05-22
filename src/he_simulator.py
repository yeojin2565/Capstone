"""
src/he_simulator.py

클라이언트 그룹별 HE latency 시뮬레이션

그룹 구성 (총 36개):
    Excellent (cid  0~ 2): N(0.02, 0.005)  극단적으로 좋은 기기
    Fast      (cid  3~12): N(0.10, 0.020)  빠른 기기
    Medium    (cid 13~22): N(0.50, 0.080)  보통 기기
    Slow      (cid 23~32): N(1.50, 0.250)  느린 기기
    Extreme   (cid 33~35): N(5.00, 0.500)  극단적으로 나쁜 기기
"""

import numpy as np

GROUP_CONFIG = {
    "excellent": {"mean": 0.02, "std": 0.005, "dropout": 0.02},
    "fast":      {"mean": 0.10, "std": 0.020, "dropout": 0.05},
    "medium":    {"mean": 0.50, "std": 0.080, "dropout": 0.10},
    "slow":      {"mean": 1.50, "std": 0.250, "dropout": 0.20},
    "extreme":   {"mean": 5.00, "std": 0.500, "dropout": 0.40},
}

TRANSMISSION_NOISE_STD = 0.01
HE_LATENCY_MAX         = 6.0   # Extreme 최대값 기준 (정규화용)


def get_group(cid: int) -> str:
    if cid < 3:    return "excellent"
    elif cid < 13: return "fast"
    elif cid < 23: return "medium"
    elif cid < 33: return "slow"
    else:          return "extreme"


def init_base_latency(cid: int, seed: int = None) -> float:
    """클라이언트 고유 HE latency 초기화 (1회)"""
    if seed is not None:
        np.random.seed(seed + cid)
    cfg  = GROUP_CONFIG[get_group(cid)]
    base = np.random.normal(cfg["mean"], cfg["std"])
    return float(np.clip(base, 0.005, HE_LATENCY_MAX))


def simulate_he_latency(base_latency: float) -> float:
    """매 라운드 HE latency = base + 전송 노이즈"""
    noise = np.random.normal(0, TRANSMISSION_NOISE_STD)
    return float(np.clip(base_latency + noise, 0.005, HE_LATENCY_MAX))


def simulate_dropout(cid: int) -> bool:
    """그룹별 dropout 확률"""
    prob = GROUP_CONFIG[get_group(cid)]["dropout"]
    return np.random.random() < prob


def normalize_latency(latency: float) -> float:
    return float(np.clip(latency / HE_LATENCY_MAX, 0.0, 1.0))