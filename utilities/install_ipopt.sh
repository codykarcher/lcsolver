#!/usr/bin/env bash
#
# The IPOPT build script moved into the package, at
# lcsolver/scripts/install_ipopt.sh, so that `lcsolver-install-solvers` can run
# it from a wheel install where utilities/ does not exist. This wrapper keeps
# the path people already type working, and forwards every argument and
# environment variable unchanged:
#
#   ./utilities/install_ipopt.sh /path/to/root
#
# See the header of the real script for MA27_SRC, IPOPT_BUILD_MUMPS, and what
# gets created where.

set -euo pipefail
exec "$(cd "$(dirname "$0")/.." && pwd)/lcsolver/scripts/install_ipopt.sh" "$@"
