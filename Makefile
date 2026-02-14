.PHONY: ui venv

VENV ?= .songviz/venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

venv:
	@test -x "$(PY)" || python3 -m venv "$(VENV)"
	@$(PIP) install -e .

ui: venv
	@$(PY) -m songviz ui
