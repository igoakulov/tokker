.PHONY: test clean build publish

test:
	PYTHONPATH=./src python -m unittest discover -v -s tests -p "test_*.py"
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .ropeproject .pytest_cache
	rm -rf .ruff_cache

build: clean
	PYTHONPATH=./src python -m build
	twine check dist/*

publish: build
	twine upload dist/*
