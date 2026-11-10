.PHONY: install test experiments figures traces serve all

install:
	pip install -r requirements.txt

test:
	pytest -q

experiments:
	python experiments/run_experiments.py

figures:
	python experiments/make_figures.py

traces:
	python experiments/export_traces.py --seed 4 13 12

serve:
	uvicorn server.app:app --reload --port 8000

all: test experiments figures traces
