#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' .env | xargs)
python roostoo_bot.py
