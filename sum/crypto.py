import tenseal as ts
import numpy as np

# ----------------------------------------
# 1. HE Context 설정
# ----------------------------------------
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60],
)

context.global_scale = 2**40
context.generate_galois_keys()


# ----------------------------------------
# 2. Weight 암호화 (chunking + scale 고정)
# ----------------------------------------
def encrypt_weights(weights, chunk_size=4096):
    """
    weights: List[np.array]
    return: List[List[ckks_vector]]
    """

    encrypted = []

    for w in weights:
        flat = w.flatten()
        chunks = []

        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i + chunk_size]

            # 핵심 수정: scale 강제 통일
            enc = ts.ckks_vector(context, chunk, scale=context.global_scale)

            chunks.append(enc)

        encrypted.append(chunks)

    return encrypted


# ----------------------------------------
# 3. Weight 복호화 (안정성 보정 포함)
# ----------------------------------------
def decrypt_weights(enc_weights, shapes):
    """
    enc_weights: List[List[ckks_vector]]
    shapes: 원래 weight shape 정보
    """

    decrypted = []

    for enc_chunks, shape in zip(enc_weights, shapes):
        flat = []

        for chunk in enc_chunks:
            values = chunk.decrypt()

            # 안정성 보정 (nan/inf 방지)
            values = np.array(values)
            values = np.nan_to_num(values, nan=0.0, posinf=1e5, neginf=-1e5)

            flat.extend(values)

        flat = np.array(flat[:np.prod(shape)])
        decrypted.append(flat.reshape(shape))

    return decrypted