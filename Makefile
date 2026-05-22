# ── 실험 실행 ──────────────────────────────────────────
# 사용법:
#   make a2c                          기본 실험
#   make random                       baseline 실험
#   make plot D=outputs/.../results.pkl R=outputs/.../results_random.pkl
#   make iid                          IID 모드로 전환
#   make noniid                       Non-IID 모드로 전환 (기본)

# ── 파라미터 (base.yaml과 동기화) ─────────────────────
ROUNDS  ?= 200
CLIENTS ?= 36
K       ?= 4

# ── 실험 실행 ──────────────────────────────────────────
.PHONY: a2c random plot iid noniid clean

a2c:
	python -m experiments.train_a2c \
		num_rounds=$(ROUNDS) \
		num_clients=$(CLIENTS) \
		num_clients_per_round_fit=$(K)

random:
	python -m experiments.train_random \
		num_rounds=$(ROUNDS) \
		num_clients=$(CLIENTS) \
		num_clients_per_round_fit=$(K)

# ── 비교 그래프 ───────────────────────────────────────
# 사용 예: make plot D=outputs/2026-05-10/.../results.pkl R=outputs/2026-05-10/.../results_random.pkl
D?=outputs/2026-05-20/23-30-31/results.pkl
R?=outputs/2026-05-21/00-14-24/results_random.pkl

plot:
	python analysis/compare_results.py \
		--dqn_path $(D) \
		--random_path $(R) \
		--save_dir analysis/figures

# ── IID / Non-IID 전환 ────────────────────────────────
iid:
	@echo "IID 모드로 전환 (alpha=100.0)"
	python -c "\
import re, pathlib; \
f = pathlib.Path('src/dataset.py'); \
f.write_text(re.sub(r'alpha=\S+,', 'alpha=100.0,', f.read_text()))"
	@echo "완료. src/dataset.py 확인하세요."

noniid:
	@echo "Non-IID 모드로 전환 (alpha=0.5)"
	python -c "\
import re, pathlib; \
f = pathlib.Path('src/dataset.py'); \
f.write_text(re.sub(r'alpha=\S+,', 'alpha=0.5,', f.read_text()))"
	@echo "완료. src/dataset.py 확인하세요."

# ── 결과 정리 ─────────────────────────────────────────
clean:
	@echo "outputs/ 폴더를 직접 삭제하세요 (실수 방지)"