"""
he_simulator.py

그룹별 HE latency 시뮬레이션 (실제 HE 연산 없음)

그룹 구성 (100 clients):
    Excellent  (cid 0~4)    : N(0.02, 0.005)  극단적으로 좋은 기기
    Fast       (cid 5~34)   : N(0.10, 0.020)  빠른 기기
    Medium     (cid 35~64)  : N(0.50, 0.080)  보통 기기
    Slow       (cid 65~89)  : N(1.50, 0.250)  느린 기기
    Extreme    (cid 90~99)  : N(5.00, 0.500)  극단적으로 나쁜 기기

수정 사항:
    [BUG-11] init_base_latency(): 독립 RNG 사용 (기존 유지)
    [BUG-12] simulate_he_latency(): 전역 np.random → cid+round 기반 독립 RNG
             Ray 병렬 환경에서 전역 랜덤 상태 오염 방지 + 재현성 보장
    [BUG-13] simulate_dropout(): 전역 np.random → cid+round 기반 독립 RNG
             동일 이유. round_num 없을 경우 cid 단독 시드 사용.
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

TRANSMISSION_NOISE_STD = 0.01
HE_LATENCY_MAX = 6.0


def get_group(cid: int) -> str:
    """클라이언트 ID → 그룹명 (100 clients 기준)"""
    if cid < 5:    return "excellent"
    elif cid < 35: return "fast"
    elif cid < 65: return "medium"
    elif cid < 90: return "slow"
    else:          return "extreme"


def init_base_latency(cid: int, seed: int = None) -> float:
    """
    클라이언트 고유 기본 HE latency 초기화 (1회)
    가우시안 분포에서 샘플링 → 클라이언트마다 고정된 성능 부여
    """
    rng    = np.random.default_rng(seed + cid if seed is not None else cid)
    group  = get_group(cid)
    config = GROUP_CONFIG[group]
    base   = rng.normal(config["mean"], config["std"])
    return float(np.clip(base, 0.005, HE_LATENCY_MAX))


def simulate_he_latency(base_latency: float, cid: int = 0, round_num: int = 0) -> float:
    """
    매 라운드 HE latency 계산: 기본 latency + 전송 중 랜덤 노이즈

    [BUG-12 FIX] 전역 np.random 대신 cid+round 기반 독립 RNG 사용.
    → Ray 병렬 환경에서 클라이언트 간 랜덤 상태 간섭 없음.
    → 동일 (cid, round_num) 조합이면 항상 동일한 노이즈 → 재현성 보장.
    """
    rng   = np.random.default_rng(cid * 100000 + round_num)
    noise = rng.normal(0, TRANSMISSION_NOISE_STD)
    return float(np.clip(base_latency + noise, 0.005, HE_LATENCY_MAX))


def simulate_dropout(cid: int, round_num: int = 0) -> bool:
    """
    그룹별 확률로 dropout 시뮬레이션
    True: 해당 라운드 탈락

    [BUG-13 FIX] 전역 np.random 대신 cid+round 기반 독립 RNG 사용.
    → he_latency RNG와 시드 충돌 방지를 위해 offset(+999983) 추가.
    """
    group = get_group(cid)
    prob  = GROUP_CONFIG[group]["dropout"]
    rng   = np.random.default_rng(cid * 100000 + round_num + 999983)
    return bool(rng.random() < prob)


def normalize_latency(latency: float) -> float:
    """reward 계산용 정규화 [0, 1]"""
    return float(np.clip(latency / HE_LATENCY_MAX, 0.0, 1.0))


# ── 단독 실행 확인 ─────────────────────────────────────
if __name__ == "__main__":
    print("── 그룹별 HE latency 분포 확인 ──\n")

    groups = {}
    for cid in range(100):
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
