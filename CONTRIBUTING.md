# Contributing to HAIIP

## Bug Reports

Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version)

## Code Contributions

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Add tests for your changes
4. Run: `pytest tests/ -v`
5. Lint: `ruff check . --select E,F,I --ignore E501,E402,F401,F841`
6. Open a pull request against `main`

## Code Style

- Python 3.11+
- ruff for linting and formatting
- Type hints on all public functions
- Docstrings for all public classes

## License

By contributing, you agree your contributions are subject to the project license.
