# Contribution Guide

## Setup
```bash
git clone https://github.com/Aliipou/HAIIP
cd HAIIP
pip install -e .[dev]
pre-commit install
```

## Running Tests
```bash
pytest tests/ -v
```

## Code Style
We use ruff for linting and formatting. Run `ruff check .` before committing.
