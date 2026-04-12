#!/usr/bin/env bash
set -euo pipefail

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required but was not found in PATH."
    exit 1
fi

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [ -n "${GENERATOR:-}" ]; then
    SELECTED_GENERATOR="${GENERATOR}"
elif command -v ninja >/dev/null 2>&1; then
    SELECTED_GENERATOR="Ninja"
else
    SELECTED_GENERATOR="Unix Makefiles"
fi

cmake -S . -B "${BUILD_DIR}" -G "${SELECTED_GENERATOR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "$@"
cmake --build "${BUILD_DIR}" --parallel
