#!/usr/bin/env bash
#
# Build IPOPT from source, against HSL MA27 or against MUMPS.
#
#   ./install_ipopt.sh /path/to/root
#
# `root` must not have a trailing slash. See docs/ipopt.rst for why MA27 rather
# than the MUMPS that every prebuilt IPOPT ships with.
#
# This script is normally driven by `lcsolver-install-solvers` (which is how it
# gets called from a wheel install, where there is no utilities/ directory),
# but the by-hand form above is unchanged and still supported.
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
# ENVIRONMENT
# -----------
#   MA27_SRC          Where the MA27 sources are, if not <root>/MA27/ma27-1.0.0.
#                     Lets the sources stay where they were extracted instead of
#                     being copied into the build root.
#   IPOPT_BUILD_MUMPS Set to 1 to build against MUMPS instead of MA27. This is
#                     the fallback for machines with no conda, apt or Homebrew
#                     to get a prebuilt IPOPT from; MUMPS is redistributable, so
#                     it needs no manual download. Requires gfortran.
#
# CREATES
# -------
#   <root>/ipopt/Ipopt           clone of the IPOPT repository
#   <root>/ipopt/ThirdParty-ASL  clone of the ASL helper
#   <root>/ipopt/ThirdParty-HSL  clone of the HSL helper       (MA27 build)
#   <root>/ipopt/ASL_build       built ASL
#   <root>/ipopt/HSL_build       built HSL, containing MA27    (MA27 build)
#   <root>/ipopt/MUMPS_build     built MUMPS                   (MUMPS build)
#   <root>/ipopt/build           built IPOPT -- bin/ipopt lives here

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 /path/to/root" >&2
    exit 2
fi

ROOT="${1%/}"                     # tolerate a trailing slash rather than break
MA27_SRC="${MA27_SRC:-$ROOT/MA27/ma27-1.0.0}"
BUILD_MUMPS="${IPOPT_BUILD_MUMPS:-0}"

if [ "$BUILD_MUMPS" != "1" ] && [ ! -d "$MA27_SRC" ]; then
    echo "error: no MA27 source at $MA27_SRC" >&2
    echo "       see the header of this script for how to obtain it," >&2
    echo "       or set IPOPT_BUILD_MUMPS=1 to build against MUMPS instead" >&2
    exit 1
fi

mkdir -p "$ROOT/ipopt"
cd "$ROOT/ipopt"

# Clone only what is missing, so the script can be re-run after a failed build
# without wiping work already done.
REPOS=(
    "https://github.com/coin-or/Ipopt.git|Ipopt"
    "https://github.com/coin-or-tools/ThirdParty-ASL.git|ThirdParty-ASL"
)
if [ "$BUILD_MUMPS" = "1" ]; then
    REPOS+=("https://github.com/coin-or-tools/ThirdParty-Mumps.git|ThirdParty-Mumps")
else
    REPOS+=("https://github.com/coin-or-tools/ThirdParty-HSL.git|ThirdParty-HSL")
fi

for repo in "${REPOS[@]}"; do
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

# --- linear solver ---------------------------------------------------------
if [ "$BUILD_MUMPS" = "1" ]; then
    # MUMPS downloads itself: it is redistributable, which is exactly why every
    # prebuilt IPOPT has it and why this path needs no manual step.
    cd "$ROOT/ipopt/ThirdParty-Mumps"
    ./get.Mumps
    mkdir -p "$ROOT/ipopt/MUMPS_build"
    ./configure --prefix="$ROOT/ipopt/MUMPS_build"
    make
    make install

    SOLVER_CONFIGURE=(
        --with-mumps-cflags="-I$ROOT/ipopt/MUMPS_build/include/coin-or/mumps"
        --with-mumps-lflags="-L$ROOT/ipopt/MUMPS_build/lib -lcoinmumps"
    )
else
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

    SOLVER_CONFIGURE=(
        --with-hsl-cflags="-I$ROOT/ipopt/HSL_build/include/coin-or/hsl"
        --with-hsl-lflags="-L$ROOT/ipopt/HSL_build/lib -lcoinhsl"
    )
fi

# --- IPOPT -----------------------------------------------------------------
cd "$ROOT/ipopt/Ipopt"
mkdir -p "$ROOT/ipopt/build"
./configure --prefix="$ROOT/ipopt/build" \
    --disable-java \
    --with-asl-cflags="-I$ROOT/ipopt/ASL_build/include/coin-or/asl" \
    --with-asl-lflags="-L$ROOT/ipopt/ASL_build/lib -lcoinasl" \
    "${SOLVER_CONFIGURE[@]}"

make
make test
make install

# --- does the binary need a library path at all? ---------------------------
# It usually does not. libtool records absolute install names for the
# ThirdParty libraries, so the loader finds them with no help. Telling every
# user to paste three exports into a shell profile "or the binary will not
# start" is cargo cult when the binary starts fine -- so test it instead of
# assuming, here, on this machine, with the variables cleared.
case "$(uname -s)" in
    Darwin) LIBVAR=DYLD_LIBRARY_PATH ;;
    *)      LIBVAR=LD_LIBRARY_PATH ;;
esac

if [ "$BUILD_MUMPS" = "1" ]; then
    SOLVER_LIB="$ROOT/ipopt/MUMPS_build/lib"
else
    SOLVER_LIB="$ROOT/ipopt/HSL_build/lib"
fi

if env -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
       "$ROOT/ipopt/build/bin/ipopt" --version >/dev/null 2>&1; then
    NEEDS_LIBPATH=0
else
    NEEDS_LIBPATH=1
fi

echo
echo "--------------------------------------------------------------------------"
echo "Build complete: $ROOT/ipopt/build/bin/ipopt"
echo

if [ "$NEEDS_LIBPATH" = "1" ]; then
    cat <<EOF
This build needs its libraries on the loader path -- it was checked just now
and would not start without them. Add to your shell profile:

export $LIBVAR="$ROOT/ipopt/build/lib:\$$LIBVAR"
export $LIBVAR="$ROOT/ipopt/ASL_build/lib:\$$LIBVAR"
export $LIBVAR="$SOLVER_LIB:\$$LIBVAR"

EOF
else
    cat <<EOF
Checked: the binary starts with no DYLD_LIBRARY_PATH or LD_LIBRARY_PATH set,
so there is nothing to add to your shell profile for it.

EOF
fi

cat <<EOF
Nothing else is required. LCsolver records this build and will use it
without any PATH change -- it prefers an MA27 build over a prebuilt MUMPS one
even when conda puts conda's first. Verify with:

    lcsolver-check-solvers

To run 'ipopt' by hand in a terminal, it does need to be findable:

export PATH="$ROOT/ipopt/build/bin:\$PATH"
--------------------------------------------------------------------------
EOF
