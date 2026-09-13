.PHONY: test research app semantic-research

test:
	python3 -m pytest -q

app:
	python3 -m streamlit run app.py

research:
	PYTHONPATH=src python3 scripts/prepare_legacy.py
	PYTHONPATH=src python3 scripts/run_research.py

semantic-research:
	python3 scripts/run_semantic_research.py
