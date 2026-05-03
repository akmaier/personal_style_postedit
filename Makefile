PY ?= python3

.PHONY: all data embeddings sims tests figures report clean distclean test mimics mimic-compare final-assessment detection embedding-viz

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

final-assessment:
	$(PY) scripts/10_final_assessment.py

detection:
	$(PY) scripts/11_detection_experiment.py

embedding-viz:
	$(PY) scripts/12_embedding_visualization.py

test:
	$(PY) -m pytest -q

# `clean` removes only files that match .gitignore (i.e. anything generated
# by the pipeline). Committed artifacts -- the LUAR mimic drafts under
# data/processed/mimics/, the figures, and the results CSVs -- are preserved.
# The previous `rm -rf data/processed figures results/*.csv` was destructive:
# it deleted the committed Opus 4.7 draft cache (324 LLM-generated texts) and
# the committed figures, ignoring the .gitignore negation rules entirely.
clean:
	@command -v git >/dev/null 2>&1 || { echo "git not available; refusing to clean"; exit 1; }
	@if [ ! -d .git ]; then echo "not a git checkout; refusing to clean"; exit 1; fi
	git clean -fdX -- data/processed figures results

# `distclean` is the explicit big-hammer version: removes BOTH ignored and
# untracked files in the same paths. Still preserves anything committed
# (so the Opus draft cache survives), but wipes any locally-staged work.
distclean:
	@command -v git >/dev/null 2>&1 || { echo "git not available; refusing to clean"; exit 1; }
	@if [ ! -d .git ]; then echo "not a git checkout; refusing to clean"; exit 1; fi
	git clean -fdx -- data/processed figures results
