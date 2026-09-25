#!/bin/bash
python -m black src tests scripts
docformatter -i -r src tests
isort src tests
