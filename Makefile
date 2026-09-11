.PHONY: test research

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

research:
	PYTHONPATH=src python3 scripts/prepare_legacy.py
	PYTHONPATH=src python3 scripts/run_research.py
