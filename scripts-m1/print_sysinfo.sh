#!/usr/bin/env bash
echo "=== System info ==="
date

# Capture real compiler and flags
echo "Compiler: $("${CXX:-g++}" --version | head -n 1 2>/dev/null || echo "unknown")"

# Capture actual build flags from Makefile
if [[ -f "Makefile" ]]; then
    echo "Makefile CXXFLAGS_BASE: $(grep '^CXXFLAGS_BASE' Makefile | sed 's/^CXXFLAGS_BASE[[:space:]]*=[[:space:]]*//')"
    echo "Makefile ANTIVEC_FLAGS: $(grep '^ANTIVEC_FLAGS' Makefile | sed 's/^ANTIVEC_FLAGS[[:space:]]*=[[:space:]]*//')"
fi

# Capture environment variables
echo "Environment CXX: ${CXX:-g++}"
echo "Environment CXXFLAGS: ${CXXFLAGS:-not set}"
echo "Environment ANTIVEC: ${ANTIVEC:-not set}"
echo "Environment DEFS: ${DEFS:-not set}"

# Capture git information
if command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null; then
    echo "Git SHA: $(git rev-parse --short HEAD)"
    echo "Git branch: $(git branch --show-current)"
    echo "Git status: $(git status --porcelain | wc -l | xargs) files modified"
else
    echo "Git: not available or not a git repository"
fi

echo "System: $(uname -a)"
(lscpu 2>/dev/null | head -n 20) || true
(free -h 2>/dev/null) || true
