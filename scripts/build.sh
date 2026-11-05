#!/usr/bin/env bash
# Build the web viewer into the Python package, then build the wheel and sdist.
# The wheel must contain jepa_scope/static/, or `pip install jepa-scope` users get the
# "viewer is not built" page.
set -euo pipefail
cd "$(dirname "$0")/.."

(cd web && npm ci && npm run build)
uv build

wheel=$(ls -t dist/*.whl | head -1)
if ! unzip -l "$wheel" | grep -q "jepa_scope/static/index.html"; then
  echo "error: $wheel does not contain the built viewer" >&2
  exit 1
fi
echo "built $wheel with the viewer included"
