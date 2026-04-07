import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ── 상수 ──────────────────────────────────────────────
N_CLIENTS   = 10    # 엣지당 클라이언트 수
N_SAMPLES   = 500   # 생성할 전체 샘플 수
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# 클라이언트 선택 레이블
# 0: 선택 안 됨 / 1: 선택됨
LABEL_MAPPING = {0: "not_selected", 1: "selected"}


def generate_dummy_data(n_sample: int = N_SAMPLES) -> pd.DataFrame:
    """더미 클라이언트 state data 생성"""
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    rows = []
    for _ in range(n_sample):
        loss          = random.uniform(0.1, 2.0)
        accuracy      = random.uniform(0.3, 0.95)
        train_latency = random.uniform(100, 800)
        he_latency    = random.uniform(50, 1000)
        data_size     = random.randint(200, 3000)
        
        # 선택 기준(좋은 클라이언트 조건)
        score = (
            (1 - loss / 2.0)        # loss 낮을 수록 유리
            + accuracy              # accuracy 높을수록 유리
            - he_latency / 1000.0   # he_latency 낮을수록 유리
            + data_size / 3000.0    # data 많을수록 유리
        )
        label = 1 if score > 1.5 else 0  # fix: 1.5만 넘기면 1로 라벨링하는 점이 조금 걸림
    
        rows.append({
            "loss":          loss,
            "accuracy":      accuracy,
            "train_latency": train_latency,
            "he_latency":    he_latency, 
            "data_size":     data_size,
            "class":         label
        })
    
    return pd.DataFrame(rows)


def load_and_preprocessing():
    """
    1. 더미 데이터 생성
    2. 전처리
    3. train / test 분리
    output: X_train, X_test, y_train, y_test
    """
    columns = [
        "loss", "accuracy", "train_latency", "he_latency", "data_size"
    ]


    # ===== 1. 더미 데이터 생성 =====
    df = generate_dummy_data(n_sample=N_SAMPLES)
    
    # ===== 2. train / test 분리 =====
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["class"],
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # ===== 3. 레이블 분리 =====
    y_train = df_train["class"]
    y_test  = df_test["class"]
    
    X_train = df_train[columns]
    X_test  = df_test[columns]
    
    # ===== 4. Min-Max 정규화 =====
    scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test) 
    
    # ===== 5. 데이터프레임 복원 =====
    X_train = pd.DataFrame(X_train_scaled, columns=columns)
    X_test  = pd.DataFrame(X_test_scaled, columns=columns)
    
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test


# ----- 단독 실행 확인 ------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocessing()
    
    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)
    print("\n레이블 분포 (train):")
    print(y_train.value_counts())
    print("\n레이블 분포 (test):")
    print(y_test.value_counts())
    print("\nX_train 샘플:")
    print(X_train.head())