#!/usr/bin/env bash

set -euo pipefail

uv run ruff check --fix src/ scripts/ tests/
uv run ruff format src/ scripts/ tests/
#uv run ty check ./src/ ./scripts/
