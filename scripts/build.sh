#!/usr/bin/env bash
# Build the web viewer into the Python package, then build the wheel and sdist.
# The wheel must contain jepa_scope/static/, or `pip install jepa-scope` users get the
# "viewer is not built" page.
set -euo pipefail
cd "$(dirname "$0")/.."

(cd web && npm ci && npm run build)
uv build

wheel=$(ls -t dist/*.whl | head -1)
# Read the listing first: piping unzip into `grep -q` fails at random under pipefail, because grep
# exits at the first match and unzip then dies writing to the closed pipe.
listing=$(unzip -l "$wheel")
if ! grep -q "jepa_scope/static/index.html" <<<"$listing"; then
  echo "error: $wheel does not contain the built viewer" >&2
  exit 1
fi
echo "built $wheel with the viewer included"
