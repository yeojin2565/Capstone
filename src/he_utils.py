"""
he_utils.py
 
TenSEAL CKKS 기반 동형암호 유틸리티
- 컨텍스트 생성 (공개키/비밀키 포함)
- 모델 가중치 암호화 / 복호화
- 암호화된 가중치 FedAvg (HE 덧셈 + 스칼라 곱)

"""
import tenseal as ts
import numpy as np
from typing import List
 
 
# ── CKKS 컨텍스트 생성 ─────────────────────────────────
def create_he_context() -> ts.Context:
    """
    CKKS 컨텍스트 생성 (공개키 + 비밀키 포함)
    서버에서 1회 생성 후 클라이언트에 공개키만 공유하는 게 이상적이나,
    시뮬레이션이므로 동일 컨텍스트를 공유해서 사용
 
    poly_modulus_degree : 8192 (보안 128-bit 수준)
    coeff_mod_bit_sizes : [60, 40, 40, 60] → 곱셈 깊이 2
    global_scale        : 2^40
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40
    return ctx
 
 
def get_public_context(ctx: ts.Context) -> ts.Context:
    """비밀키 제거한 공개 컨텍스트 반환 (클라이언트 배포용)"""
    pub_ctx = ctx.copy()
    pub_ctx.make_context_public()
    return pub_ctx
 
 
# ── 가중치 암호화 ──────────────────────────────────────
def encrypt_weights(
    weights: List[np.ndarray],
    ctx: ts.Context,
) -> List[ts.CKKSVector]:
    """
    모델 가중치 리스트 → CKKSVector 리스트
 
    각 레이어 가중치를 flatten → CKKSVector로 암호화
    복호화 후 원래 shape로 복원하려면 shape 정보 별도 저장 필요
    """
    encrypted = []
    for w in weights:
        flat = w.flatten().astype(np.float64)
        enc  = ts.ckks_vector(ctx, flat.tolist())
        encrypted.append(enc)
    return encrypted
 
 
# ── 가중치 복호화 ──────────────────────────────────────
def decrypt_weights(
    encrypted: List[ts.CKKSVector],
    shapes: List[tuple],
) -> List[np.ndarray]:
    """
    CKKSVector 리스트 + 원래 shape → numpy 가중치 리스트 복원
    """
    decrypted = []
    for enc, shape in zip(encrypted, shapes):
        flat = np.array(enc.decrypt(), dtype=np.float32)
        decrypted.append(flat.reshape(shape))
    return decrypted
 
 
# ── HE FedAvg (암호화 상태로 평균) ────────────────────
def he_fedavg(
    all_encrypted: List[List[ts.CKKSVector]],
    n_clients: int,
) -> List[ts.CKKSVector]:
    """
    암호화된 가중치들을 HE 덧셈으로 합산 후 스칼라 나눗셈
 
    all_encrypted : [client1_enc_weights, client2_enc_weights, ...]
    n_clients     : 참여 클라이언트 수
    반환          : 평균 암호화 가중치 리스트
    """
    # 레이어별 합산
    aggregated = [enc.copy() for enc in all_encrypted[0]]
 
    for client_weights in all_encrypted[1:]:
        for i, enc_layer in enumerate(client_weights):
            aggregated[i] += enc_layer   # HE 덧셈
 
    # 스칼라 나눗셈 (1/n)
    averaged = [enc * (1.0 / n_clients) for enc in aggregated]
 
    return averaged