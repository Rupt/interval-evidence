SHELL := /bin/bash


.PHONY: help
help:
	@echo "usage:"
	@echo "make test  # run tests"
	@echo "make fmt   # format code"
	@echo "make clean # clean up"


.PHONY: test
test:
	python test.py


.PHONY: fmt
fmt:
	isort lebesgue/*.py *.py --profile black
	black lebesgue/*.py *.py


.PHONY: clean
clean:
	rm -rf __pycache__ lebesgue/__pycache__


env_lebesgue/bin/activate:
	python3 -m venv env_lebesgue
	( \
	source env_lebesgue/bin/activate; \
	pip install --upgrade pip; \
	pip install scipy numpy numba black isort; \
	)
