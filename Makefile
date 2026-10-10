PY ?= python3
export PYTHONPATH := src

.PHONY: test env audit split

test:
	$(PY) -m unittest discover -s tests -v

env:
	$(PY) scripts/00_record_environment.py

# Kullanım: make audit ZIP=/yol/InSDN_DatasetCSV.zip
audit:
	$(PY) scripts/01_audit_data.py --zip $(ZIP)

split:
	$(PY) scripts/02_make_split.py --zip $(ZIP)
