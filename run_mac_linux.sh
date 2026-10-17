#!/usr/bin/env bash
# Starts NeuroScreen. The browser opens at http://localhost:8000
cd "$(dirname "$0")" || exit 1
python3 -m pip install -r requirements.txt
python3 app.py
