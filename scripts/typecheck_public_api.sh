#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PUBLIC_API_MODULES=(
  "genlib/__init__.py"
  "src/__init__.py"
  "src/compat/__init__.py"
  "src/configs/__init__.py"
  "src/core/__init__.py"
  "src/datasets/__init__.py"
  "src/losses/__init__.py"
  "src/models/__init__.py"
  "src/nn/__init__.py"
  "src/noise/__init__.py"
  "src/pipelines/__init__.py"
  "src/plugins/__init__.py"
  "src/sampling/__init__.py"
  "src/scheduling/__init__.py"
  "src/training/__init__.py"
  "src/utils/__init__.py"
)

".venv/bin/python" -m mypy --follow-imports=silent "${PUBLIC_API_MODULES[@]}"
