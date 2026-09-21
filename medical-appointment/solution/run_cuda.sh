#!/usr/bin/env bash
# Run from the challenge directory with the active Python environment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export DEVICE="${DEVICE:-cuda:0}"
cuda_libraries=$(python - <<'PY'
from pathlib import Path
import nvidia.cublas
import nvidia.cudnn

packages = (nvidia.cublas, nvidia.cudnn)
print(":".join(str(Path(next(iter(package.__path__))) / "lib") for package in packages))
PY
)
# CTranslate2 resolves these shared libraries when the process starts.
export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [ "$#" -eq 0 ]; then
    set -- -m solution.solution
fi
exec python "$@"
