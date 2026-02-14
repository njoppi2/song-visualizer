.PHONY: ui venv

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

venv:
	@test -x "$(PY)" || python3 -m venv "$(VENV)"
	@$(PIP) install -e .

ui: venv
	@$(PY) -m songviz ui

