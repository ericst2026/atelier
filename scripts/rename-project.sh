#!/usr/bin/env bash
# Rename the project everywhere: display name, Python package, env prefix, Redis
# keys, Prometheus metrics, container names and server paths.
#
#   ./scripts/rename-project.sh Studio studio
#   ./scripts/rename-project.sh 공방 gongbang
#
# Arg 1 is the display name people see; arg 2 is the lowercase identifier used in
# code and paths (letters, digits and underscores only — it becomes a Python
# package name). Run this on a fresh checkout, before the first `docker compose up`.
set -euo pipefail

DISPLAY="${1:-}"
SLUG="${2:-}"
if [[ -z "$DISPLAY" || -z "$SLUG" ]]; then
  echo "usage: $0 <DisplayName> <lowercase_slug>" >&2
  exit 1
fi
if [[ ! "$SLUG" =~ ^[a-z][a-z0-9_]*$ ]]; then
  echo "the slug must start with a letter and contain only a-z, 0-9 and _" >&2
  exit 1
fi
UPPER=$(echo "$SLUG" | tr '[:lower:]' '[:upper:]')

cd "$(dirname "$0")/.."
if [[ -d .dev || -d data ]]; then
  echo "warning: this rewrites env var names and Redis keys. Existing runs in .dev/ or data/ keep working, but a running stack must be restarted." >&2
fi

# Files to touch: source, config and docs — never binaries or dependencies.
mapfile -t FILES < <(git ls-files 2>/dev/null || find . \
  -path ./node_modules -prune -o -path ./.git -prune -o -path ./frontend/dist -prune -o \
  -name '__pycache__' -prune -o -type f \
  \( -name '*.py' -o -name '*.js' -o -name '*.jsx' -o -name '*.css' -o -name '*.html' \
     -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' -o -name '*.sh' \
     -o -name '*.txt' -o -name '.env.example' -o -name 'Dockerfile*' -o -name '.gitignore' \) -print)

echo "rewriting ${#FILES[@]} files…"
for f in "${FILES[@]}"; do
  # order matters: the all-caps env prefix first, then the display name, then the slug
  sed -i \
    -e "s/ATELIER_/${UPPER}_/g" \
    -e "s/Atelier/${DISPLAY}/g" \
    -e "s/atelier/${SLUG}/g" \
    "$f"
done

# The header wordmark splits the name for the two-tone effect; put it back together.
for f in frontend/src/components/TopBar.jsx frontend/src/pages/Login.jsx; do
  python3 - "$f" "$DISPLAY" <<'PY'
import re, sys
path, name = sys.argv[1], sys.argv[2]
s = open(path, encoding="utf-8").read()
head, tail = (name[:len(name)//2] or name), name[len(name)//2:]
s = re.sub(r"Ate<b>lier</b>", f"{head}<b>{tail}</b>", s)
open(path, "w", encoding="utf-8").write(s)
PY
done

# Finally the Python packages: the backend app and the two experiment libraries
# (their import names were already rewritten above, so the folders must follow).
[[ -d backend/atelier ]] && mv backend/atelier "backend/${SLUG}"
[[ -d experiments/_lib/atelier_sdk ]] && mv experiments/_lib/atelier_sdk "experiments/_lib/${SLUG}_sdk"
[[ -d experiments/_lib/atelier_nlp ]] && mv experiments/_lib/atelier_nlp "experiments/_lib/${SLUG}_nlp"

echo
echo "done. Check the result:"
echo "  grep -ril atelier . | grep -v node_modules   # should print nothing"
echo "  cd backend && python tests/test_core.py"
echo
echo "Then rename the folder itself if you like:  mv \"\$(pwd)\" ../${SLUG}"
