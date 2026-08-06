#!/usr/bin/env bash
#
# Build IPOPT with the HSL MA27 linear solver.
#
#   ./tools/install_ipopt.sh /path/to/root
#
# `root` must not have a trailing slash. See docs/ipopt.rst for why MA27 rather
# than the MUMPS that every prebuilt IPOPT ships with.
#
# BEFORE RUNNING
# --------------
# MA27 is free for academic use but cannot be redistributed, so it has to be
# fetched by hand:
#
#   1. Register and accept the licence at
#      https://www.hsl.rl.ac.uk/download/MA27/1.0.0/
#   2. Download and extract the archive to <root>/MA27/ma27-1.0.0
#
# Without that directory the HSL build produces a library with no solver in it
# and IPOPT rejects `linear_solver ma27` at run time, which is a confusing way
# to find out. This script checks for it and stops.
#
# CREATES
# -------
#   <root>/ipopt/Ipopt           clone of the IPOPT repository
#   <root>/ipopt/ThirdParty-ASL  clone of the ASL helper
#   <root>/ipopt/ThirdParty-HSL  clone of the HSL helper
#   <root>/ipopt/ASL_build       built ASL
#   <root>/ipopt/HSL_build       built HSL, containing MA27
#   <root>/ipopt/build           built IPOPT -- bin/ipopt lives here

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 /path/to/root" >&2
    exit 2
fi

ROOT="${1%/}"                     # tolerate a trailing slash rather than break
MA27_SRC="$ROOT/MA27/ma27-1.0.0"

if [ ! -d "$MA27_SRC" ]; then
    echo "error: no MA27 source at $MA27_SRC" >&2
    echo "       see the header of this script for how to obtain it" >&2
    exit 1
fi

mkdir -p "$ROOT/ipopt"
cd "$ROOT/ipopt"

# Clone only what is missing, so the script can be re-run after a failed build
# without wiping work already done.
for repo in \
    "https://github.com/coin-or/Ipopt.git|Ipopt" \
    "https://github.com/coin-or-tools/ThirdParty-ASL.git|ThirdParty-ASL" \
    "https://github.com/coin-or-tools/ThirdParty-HSL.git|ThirdParty-HSL"
do
    url="${repo%|*}"; dir="${repo#*|}"
    [ -d "$dir" ] || git clone "$url" "$dir"
done

# --- ASL -------------------------------------------------------------------
cd "$ROOT/ipopt/ThirdParty-ASL"
./get.ASL
mkdir -p "$ROOT/ipopt/ASL_build"
./configure --prefix="$ROOT/ipopt/ASL_build"
make
make install

# --- HSL / MA27 ------------------------------------------------------------
# ThirdParty-HSL expects the sources under coinhsl/, with MA27's `src`
# directory renamed to `ma27`.
cd "$ROOT/ipopt/ThirdParty-HSL"
rm -rf coinhsl
mkdir -p coinhsl
cp -a "$MA27_SRC/." coinhsl/
[ -d coinhsl/src ] && mv coinhsl/src coinhsl/ma27

mkdir -p "$ROOT/ipopt/HSL_build"
./configure --prefix="$ROOT/ipopt/HSL_build"
make
make install

# --- IPOPT -----------------------------------------------------------------
cd "$ROOT/ipopt/Ipopt"
mkdir -p "$ROOT/ipopt/build"
./configure --prefix="$ROOT/ipopt/build" \
    --disable-java \
    --with-asl-cflags="-I$ROOT/ipopt/ASL_build/include/coin-or/asl" \
    --with-asl-lflags="-L$ROOT/ipopt/ASL_build/lib -lcoinasl" \
    --with-hsl-cflags="-I$ROOT/ipopt/HSL_build/include/coin-or/hsl" \
    --with-hsl-lflags="-L$ROOT/ipopt/HSL_build/lib -lcoinhsl"

make
make test
make install

# --- what to put in the shell profile --------------------------------------
# Printed rather than appended: this script should not silently edit a profile,
# and on Linux the library variable is LD_LIBRARY_PATH instead.
case "$(uname -s)" in
    Darwin) LIBVAR=DYLD_LIBRARY_PATH ;;
    *)      LIBVAR=LD_LIBRARY_PATH ;;
esac

cat <<EOF

--------------------------------------------------------------------------
Build complete: $ROOT/ipopt/build/bin/ipopt

Add to your shell profile (~/.zshenv, ~/.bashrc). The library path is not
optional -- without it the binary exists and will not start.

export PATH="$ROOT/ipopt/build/bin:\$PATH"
export $LIBVAR="$ROOT/ipopt/build/lib:\$$LIBVAR"
export $LIBVAR="$ROOT/ipopt/ASL_build/lib:\$$LIBVAR"
export $LIBVAR="$ROOT/ipopt/HSL_build/lib:\$$LIBVAR"
export IPOPT_INCLUDE_DIR="$ROOT/ipopt/build/include/coin-or"
export IPOPT_LIBRARY_DIR="$ROOT/ipopt/build/lib"

Export the last two BEFORE 'pip install cyipopt' if you want the in-process
route to build against this IPOPT rather than a separate conda one.

Then check it:   ipopt --version
--------------------------------------------------------------------------
EOF
