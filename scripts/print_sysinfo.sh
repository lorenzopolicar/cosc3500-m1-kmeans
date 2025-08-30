#!/usr/bin/env bash
echo "=== System info ==="
date
echo "Compiler: $("${CXX:-g++}" --version | head -n 1 2>/dev/null || echo "unknown")"
echo "Flags: ${CXXFLAGS:-"(default)"}"
uname -a
(lscpu 2>/dev/null | head -n 20) || true
(free -h 2>/dev/null) || true
