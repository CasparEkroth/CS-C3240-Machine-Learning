

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

format: install
	.venv/bin/black .

lint: install
	.venv/bin/ruff check .

test: install
	.venv/bin/python3 -m pytest tests/
