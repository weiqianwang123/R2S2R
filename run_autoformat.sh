#!/bin/bash
python -m black src tests
docformatter -i -r src tests
isort src tests
