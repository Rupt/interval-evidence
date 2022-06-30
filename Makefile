SHELL := /bin/bash
PYFILES := {lebesgue,test,discohist,searches/*/}/*.py


.PHONY: help
help:
	@echo "usage:"
	@echo "make test  # run tests"
	@echo "make bench # time calls"
	@echo "make fmt   # format and lint code"
	@echo "make clean # clean up"


.PHONY: test
test:
	NUMBA_DISABLE_JIT=1 python test.py
	python test.py


.PHONY: bench
bench:
	python -m timeit -s "from benchmark import model_integrate_1 as b" "b()"


.PHONY: fmt
fmt:
	isort $(PYFILES) --profile black --line-length 79
	black $(PYFILES) -l79
	flake8 $(PYFILES); :


.PHONY: clean
clean:
	rm -rf __pycache__ lebesgue/__pycache__


env/bin/activate:
	python3 -m venv env
	( \
	source env/bin/activate; \
	pip install --upgrade pip; \
	pip install scipy numpy numba black isort flake8; \
	pip install pyhf==0.7.0rc1 cabinetry==0.4.1 --no-dependencies; \
	pip install jax jaxlib; \
	)
