"""
- 실제 HE 연산 없음
- 그룹별 HE latency 시뮬레이션

그룹 구성 (36 clients):
    Excellent  (cid 0~2)   : N(0.02, 0.005)  극단적으로 좋은 기기
    Fast       (cid 3~12)  : N(0.10, 0.020)  빠른 기기
    Medium     (cid 13~22) : N(0.50, 0.080)  보통 기기
    Slow       (cid 23~32) : N(1.50, 0.250)  느린 기기
    Extreme    (cid 33~35) : N(5.00, 0.500)  극단적으로 나쁜 기기
"""

import numpy as np


# ── 그룹 설정 ──────────────────────────────────────────
GROUP_CONFIG = {
    "excellent": {"mean": 0.02, "std": 0.005, "dropout": 0.02},
    "fast":      {"mean": 0.10, "std": 0.020, "dropout": 0.05},
    "medium":    {"mean": 0.50, "std": 0.080, "dropout": 0.10},
    "slow":      {"mean": 1.50, "std": 0.250, "dropout": 0.20},
    "extreme":   {"mean": 5.00, "std": 0.500, "dropout": 0.40},
}


# ── 전송 노이즈 ────────────────────────────────────────
TRANSMISSION_NOISE_STD = 0.01   # 매 라운드 전송 중 발생하는 랜덤 노이즈


# ── Reward 정규화 기준 ─────────────────────────────────
HE_LATENCY_MAX = 6.0            # Extreme 최대값 기준


def get_group(cid: int) -> str:
    """클라이언트 ID -> 그룹명"""
    if cid < 3:         return "excellent"
    elif cid < 13:      return "fast"
    elif cid < 23:      return "medium"
    elif cid < 33:      return "slow"
    else:               return "extreme"


def init_base_latency(cid: int, seed: int = None) -> float:
    """
    클라이언트 고유 기본 HE latency 초기화 (1회)
    가우시안 분포에서 샘플링 -> 클라이언트마다 고정된 성능 부여
    """
    if seed is not None:
        np.random.seed(seed + cid)
        
    group  = get_group(cid)
    config = GROUP_CONFIG[group]   # mean, std, dropout
    base = np.random.normal(config["mean"], config["std"]) # HE latency(size: 1개의 스칼라값)
    
    return float(np.clip(base, 0.005, HE_LATENCY_MAX)) # [0.05, MAX] clipping
    

def simulate_he_latency(base_latency: float) -> float:
    """
    매 라운드 HE latency 계산
    기본 latency + 전송 중 랜덤 노이즈
    """
    noise   = np.random.normal(0, TRANSMISSION_NOISE_STD)
    latency = base_latency + noise
    return float(np.clip(latency, 0.005, HE_LATENCY_MAX))


def simulate_dropout(cid: int) -> bool:
    """
    HE latency 기반 dropout 시뮬레이션
    True: 해당 라운드 탈락
    """
    group = get_group(cid)
    prob  = GROUP_CONFIG[group]["dropout"] # dropout probability
    return np.random.random() < prob # True일 경우 라운드 탈락
 
 
def normalize_latency(latency: float) -> float:
    """reward 계산용 정규화 [0, 1]"""
    return float(np.clip(latency / HE_LATENCY_MAX, 0.0, 1.0))


# ── 단독 실행 확인 ─────────────────────────────────────
if __name__ == "__main__":
    print("── 그룹별 HE latency 분포 확인 ──\n")
 
    groups = {}
    for cid in range(36): # [fix]: hard-coding
        group = get_group(cid)
        base  = init_base_latency(cid, seed=42)
        if group not in groups:
            groups[group] = []
        groups[group].append(base)
 
    for group, latencies in groups.items():
        print(
            f"{group:10s} | "
            f"n={len(latencies):2d} | "
            f"mean={np.mean(latencies):.3f}s | "
            f"min={np.min(latencies):.3f}s | "
            f"max={np.max(latencies):.3f}s"
        )