#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"

if ! command -v ctest >/dev/null 2>&1; then
    echo "ctest is required but was not found in PATH."
    exit 1
fi

if [ ! -d "${BUILD_DIR}" ]; then
    echo "Build directory '${BUILD_DIR}' does not exist. Run ./scripts/build.sh first."
    exit 1
fi

ctest --test-dir "${BUILD_DIR}" --output-on-failure

