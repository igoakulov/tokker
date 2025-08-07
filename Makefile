.PHONY: test clean build publish ruff

test:
	PYTHONPATH=./src python -m unittest discover -v -s tests -p "test_*.py"

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .ropeproject .pytest_cache
	rm -rf .ruff_cache
	@echo "REMINDER: Update version in pyproject.toml before building."

build: clean
	PYTHONPATH=./src python -m build
	twine check dist/*

publish: build
	twine upload dist/*

ruff:
	@command -v ruff >/dev/null 2>&1 || { echo "ruff not found. Install with: pip install ruff"; exit 1; }
	ruff check --fix .
	ruff format .
