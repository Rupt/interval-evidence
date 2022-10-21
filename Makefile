SHELL := /bin/bash
PYFILES := {lebesgue,discohisto,test,searches/*/,report}/*.py


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
	python -m venv env
	( \
	source env/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt \
	)
