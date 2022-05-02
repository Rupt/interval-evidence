# interval-evidence
Calculate probabilities with interval methods.

## setup

TODO makefile

Virtual environment

```bash
python3 -m venv env_lebesgue
source env_lebesgue/bin/activate

pip install --upgrade pip
pip install scipy numpy numba numba-scipy black isort

```


## development


```bash
black lebesgue/*.py
isort lebesgue/*.py

```


## clean

```bash
rm -r env_lebesgue

```


## test

```bash
python test.py

```
