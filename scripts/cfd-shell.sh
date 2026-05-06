#!/usr/bin/env bash
# Drop into an interactive OpenFOAM shell inside the ApexFlow CFD container.
# Run from anywhere; resolves repo root from the script location.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/cfd/docker"

# Pass host UID/GID so files created in the container are owned by you on macOS.
# UID and GID are read-only built-ins in bash, so use HOST_UID/HOST_GID.
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

docker compose run --rm cfd
