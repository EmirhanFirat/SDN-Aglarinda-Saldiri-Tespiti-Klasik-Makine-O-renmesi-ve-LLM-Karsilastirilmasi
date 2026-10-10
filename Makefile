PY ?= python3
export PYTHONPATH := src

.PHONY: test env audit split train test-eval figures

test:
	$(PY) -m unittest discover -s tests -v

env:
	$(PY) scripts/00_record_environment.py

# Kullanım: make audit DATA=/yol/InSDN_DatasetCSV   (klasör veya .zip)
audit:
	$(PY) scripts/01_audit_data.py --data $(DATA)

split:
	$(PY) scripts/02_make_split.py --data $(DATA)

train:
	$(PY) scripts/03_train_eval.py --data $(DATA)

test-eval:
	$(PY) scripts/03_train_eval.py --data $(DATA) --evaluate-test

figures:
	$(PY) scripts/04_make_figures.py
