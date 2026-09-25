#!/bin/bash
set -e
./run_autoformat.sh
mypy src tests
pytest src tests --pylint -m pylint --pylint-rcfile=.pylintrc
pytest tests/
