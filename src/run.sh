#/bin/bash

uv run train.py
uv run uvicorn api.app:app --host "0.0.0.0"