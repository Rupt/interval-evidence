SHELL := /bin/bash


.PHONY: help
help:
	@echo "make test  # run tests"
	@echo "make fmt   # autoformat code"
	@echo "make clean # clean up"


.PHONY: test
test:
	NUMBA_DISABLE_JIT=1 python test.py


.PHONY: fmt
fmt:
	isort lebesgue/*.py *.py
	black lebesgue/*.py *.py


.PHONY: clean
clean:
	git clean -fdx


env_lebesgue/bin/activate:
	python3 -m venv env_lebesgue
	( \
	source env_lebesgue/bin/activate; \
	pip install --upgrade pip; \
	pip install scipy numpy numba numba-scipy black isort; \
	)
