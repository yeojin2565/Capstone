"""pkl_to_csv.py"""

import pickle
import pandas as pd
from pathlib import Path


def find_latest_pkl(filename: str = "results.pkl",
                    base_dir: str = "outputs") -> Path:
    """outputs/ 하위에서 가장 최근 pkl 자동 탐색"""
    candidates = sorted(Path(base_dir).rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"{filename} 없음")
    return candidates[-1]  # 경로명 정렬 = 시간순

# random 돌린 후에는 results_random.pkl로 고치기
pkl_path = find_latest_pkl("results.pkl")
print(f"불러온 파일: {pkl_path}")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame(data["dqn_metrics"])
df.to_csv(pkl_path.parent / "dqn_metrics.csv", index=False)
print(f"저장 완료: {pkl_path.parent / 'dqn_metrics.csv'}")