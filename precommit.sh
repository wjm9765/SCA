#!/usr/bin/env bash

set -euo pipefail

uv run ruff check --fix ./src/ ./scripts/
uv run ruff format ./src/ ./scripts/
#uv run ty check ./src/ ./scripts/
