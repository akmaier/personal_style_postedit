PY ?= python3

.PHONY: all data embeddings sims tests figures report clean test mimics mimic-compare

all: data embeddings sims tests figures report

data:
	$(PY) scripts/00_build_dataframe.py

embeddings:
	$(PY) scripts/01_compute_embeddings.py

sims:
	$(PY) scripts/02_compute_similarities.py

tests:
	$(PY) scripts/03_run_hypothesis_tests.py

figures:
	$(PY) scripts/05_make_figures.py

report:
	$(PY) scripts/06_build_report.py

mimics:
	$(PY) scripts/07_generate_mimics.py --generators gpt-5.5 claude-opus-4-7

mimic-compare:
	$(PY) scripts/08_compare_mimics.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/processed figures results/*.csv results/*.md
