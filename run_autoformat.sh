#!/bin/bash
python -m black src tests scripts
isort src tests scripts
