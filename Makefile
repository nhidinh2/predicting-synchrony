.PHONY: install pipeline backtest dashboard test clean

install:
	pip install -e ".[dev]"
	cd dashboard && npm install

pipeline:
	python -m scripts.run_pipeline --config pipeline/config.py

backtest:
	python -m scripts.run_backtest --holdout 2024-08

dashboard:
	cd dashboard && npm run dev

build-dashboard:
	cd dashboard && npm run build

test:
	pytest tests/ -v

lint:
	ruff check pipeline/ scripts/ tests/

clean:
	rm -rf data/interim/* data/processed/*
	find . -type d -name __pycache__ -exec rm -rf {} +
