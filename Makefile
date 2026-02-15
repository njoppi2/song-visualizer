.PHONY: ui venv

VENV ?= .songviz/venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
UI_LAYOUT ?= stems4

venv:
	@test -x "$(PY)" || python3 -m venv "$(VENV)"
	@$(PIP) install -e '.[stems]'

ui: venv
	@$(PY) -m songviz ui --layout $(UI_LAYOUT)
