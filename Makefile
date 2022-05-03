SHELL := /bin/bash


.PHONY: help
help:
	@echo "usage:"
	@echo "make test  # run tests"
	@echo "make bench # time calls"
	@echo "make fmt   # format and lint code"
	@echo "make clean # clean up"


.PHONY: test
test:
	python test.py


.PHONY: bench
bench:
	python -m timeit -s "from benchmark import model_integrate_1 as b" "b()"


.PHONY: fmt
fmt:
	isort lebesgue/*.py *.py
	black lebesgue/*.py *.py
	flake8 lebesgue/ *.py; :


.PHONY: clean
clean:
	rm -rf __pycache__ lebesgue/__pycache__


env_lebesgue/bin/activate:
	python3 -m venv env_lebesgue
	( \
	source env_lebesgue/bin/activate; \
	pip install --upgrade pip; \
	pip install scipy numpy numba black isort flake8; \
	)
