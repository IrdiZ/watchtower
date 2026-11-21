#!/usr/bin/env sh
# Full test suite in the project image, from a copy of this checkout (agent worktrees excluded).
# A failure is rerun on its own: passes alone -> FLAKY (load-dependent), fails again -> FAIL (exit 1).
# Usage (Git Bash, from the repo root):  sh bench/test.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd -W 2>/dev/null || pwd)
export MSYS_NO_PATHCONV=1
docker run --rm -v "$SRC:/src:ro" --entrypoint sh watchtower -c '
  cp -r /src /tmp/w && rm -rf /tmp/w/.claude && cd /tmp/w
  pip install -q -r requirements-dev.txt >/dev/null 2>&1
  python -m pytest -q -p no:cacheprovider -rf 2>&1 | tee /tmp/run.txt | tail -4
  FAILED=$(grep "^FAILED " /tmp/run.txt | sed "s/^FAILED \([^ ]*\).*/\1/")
  [ -z "$FAILED" ] && { echo "RESULT: PASS"; exit 0; }
  RC=0
  for t in $FAILED; do
    if python -m pytest -q -p no:cacheprovider "$t" >/dev/null 2>&1; then echo "FLAKY (passes alone): $t"
    else echo "FAIL: $t"; RC=1; fi
  done
  [ $RC = 0 ] && echo "RESULT: PASS (flaky tests only)" || echo "RESULT: FAIL"
  exit $RC'
