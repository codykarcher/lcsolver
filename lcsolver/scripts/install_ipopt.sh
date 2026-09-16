#!/usr/bin/env bash
#
# Build IPOPT from source with the open-source linear solvers -- MUMPS and
# SPRAL -- and, when asked, HSL MA27 as well.
#
#   ./install_ipopt.sh /path/to/root                   MUMPS + SPRAL
#   ./install_ipopt.sh /path/to/root --with-ma27 SRC   ... plus MA27
#   ./install_ipopt.sh /path/to/root --add-ma27 SRC    add MA27 to a root
#                                                      built earlier
#
# Every solver in the build is switchable per solve (lcsolver.solve(f,
# linear_solver='spral')); with no MA27 LCsolver defaults to SPRAL, with
# MA27 to MA27. See docs/linear_solvers.rst for the measurements behind that.
#
# This script is normally driven by `lcsolver-install-solvers` (which is how
# it gets called from a wheel install, where there is no utilities/
# directory), but the by-hand form above is supported.
#
# WHAT GETS BUILT, AND WHY TWO IPOPTS
# -----------------------------------
# SPRAL is OpenMP code built with GCC (libgomp). A conda Python already holds
# LLVM's OpenMP runtime (libomp), and two OpenMP runtimes in one process is a
# hard abort -- so SPRAL cannot live in the shared library cyipopt loads
# in-process. Two products from one source, then:
#
#   <root>/ipopt/build/bin/ipopt   STATIC executable: MUMPS + SPRAL [+ MA27]
#   <root>/ipopt/build/lib/        SHARED library for cyipopt: MUMPS [+ MA27]
#
# cyipopt only evaluates grey-box functions in-process and never needs to
# pick the linear solver, so it loses nothing. Without a GCC toolchain (or
# METIS/hwloc) SPRAL is skipped with a message, and one shared build serves
# both products.
#
# The build is idempotent per component: a component whose library exists
# in the root is not rebuilt, so a failed run can be re-run, and --add-ma27
# reuses everything except the IPOPT link itself.
#
# MA27
# ----
# MA27 is free for academic use but cannot be redistributed, so it has to be
# fetched by hand:
#   1. Register and accept the licence at
#      https://www.hsl.rl.ac.uk/download/MA27/1.0.0/
#   2. Download and extract; pass the extracted directory as SRC above.
#
# OPTIONS
# -------
#   --with-ma27 SRC   include MA27 (sources at SRC) in this build
#   --add-ma27 SRC    build the HSL component into an existing root and
#                     relink both IPOPT products with it
#   --no-spral        skip SPRAL even if the toolchain allows it
#   --no-mumps        skip MUMPS (then MA27 or SPRAL must be present)
#
# ENVIRONMENT
# -----------
#   MA27_SRC          same as --with-ma27 SRC (kept for older callers)
#   JOBS              make parallelism (default 4 -- a laptop-safe choice)
#   IPOPT_BUILD_MUMPS accepted and ignored: MUMPS is now always built
#
# CREATES
# -------
#   <root>/ipopt/Ipopt             IPOPT source (stable/3.14)
#   <root>/ipopt/ThirdParty-ASL    ASL helper           -> ASL_build
#   <root>/ipopt/ThirdParty-Mumps  MUMPS helper         -> MUMPS_build
#   <root>/ipopt/ThirdParty-HSL    HSL helper           -> HSL_build   (MA27)
#   <root>/ipopt/spral             SPRAL source         -> SPRAL_build
#   <root>/ipopt/Ipopt-build-exe   static-exe build tree (wiped each run)
#   <root>/ipopt/Ipopt-build-lib   shared-lib build tree (wiped each run)
#   <root>/ipopt/build             the install -- bin/ipopt, lib/, pkgconfig

# no -u: macOS ships bash 3.2, where expanding an EMPTY array (a solver
# left out of the build) is "unbound variable" under set -u
set -eo pipefail

usage() {
    echo "usage: $0 /path/to/root [--with-ma27 SRC | --add-ma27 SRC] [--no-spral] [--no-mumps]" >&2
    exit 2
}

[ $# -ge 1 ] || usage
ROOT="${1%/}"; shift
MA27_SRC="${MA27_SRC:-}"
ADD_MA27=0
WANT_SPRAL=1
WANT_MUMPS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --with-ma27) MA27_SRC="$2"; shift 2 ;;
        --add-ma27)  MA27_SRC="$2"; ADD_MA27=1; shift 2 ;;
        --no-spral)  WANT_SPRAL=0; shift ;;
        --no-mumps)  WANT_MUMPS=0; shift ;;
        *) usage ;;
    esac
done
JOBS="${JOBS:-4}"
IP="$ROOT/ipopt"

if [ -n "$MA27_SRC" ] && [ ! -d "$MA27_SRC" ]; then
    echo "error: no MA27 source directory at $MA27_SRC" >&2
    echo "       see the header of this script for how to obtain it" >&2
    exit 1
fi
if [ "$ADD_MA27" = "1" ] && [ ! -d "$IP/Ipopt" ]; then
    echo "error: --add-ma27 needs a root built by this script; $IP/Ipopt is missing" >&2
    exit 1
fi
if [ "$WANT_MUMPS" = "0" ] && [ -z "$MA27_SRC" ] && [ "$WANT_SPRAL" = "0" ]; then
    echo "error: nothing to build IPOPT against (no MUMPS, no SPRAL, no MA27)" >&2
    exit 1
fi

say() { echo; echo "==> $*"; }

# --- toolchain -------------------------------------------------------------
# SPRAL and IPOPT must share ONE C++ runtime: SPRAL is GCC-only (OpenMP in
# Fortran = libgomp), so when SPRAL is built, IPOPT is built with the same
# GCC. Versioned names first: on macOS `gcc` is clang in disguise.
GCC_VER=""
for v in 16 15 14 13 12 11 ""; do
    suffix="${v:+-$v}"
    if command -v "gcc$suffix" >/dev/null 2>&1 \
       && command -v "g++$suffix" >/dev/null 2>&1 \
       && command -v "gfortran$suffix" >/dev/null 2>&1 \
       && "gcc$suffix" --version 2>/dev/null | head -1 | grep -qi "gcc"; then
        GCC_VER="$suffix"
        break
    fi
done
if [ -n "$GCC_VER" ]; then
    GCC_CC="gcc$GCC_VER"; GCC_CXX="g++$GCC_VER"; GCC_FC="gfortran$GCC_VER"
else
    GCC_CC=""; GCC_CXX=""; GCC_FC=""
fi
FC_ANY="$(command -v gfortran || command -v "gfortran$GCC_VER" 2>/dev/null || true)"

case "$(uname -s)" in
    Darwin) OS=mac; LIBVAR=DYLD_LIBRARY_PATH; SO=dylib ;;
    *)      OS=linux; LIBVAR=LD_LIBRARY_PATH; SO=so ;;
esac

# SPRAL's extra dependencies: METIS, hwloc, a BLAS/LAPACK, autotools.
SPRAL_WHY=""
BLAS_LFLAGS=""; METIS_LFLAGS=""; METIS_INC=""; HWLOC_LFLAGS=""
if [ "$WANT_SPRAL" = "1" ]; then
    if [ -z "$GCC_CC" ]; then
        SPRAL_WHY="no GCC toolchain (gcc, g++ and gfortran of one version) -- SPRAL is GCC-only"
    elif ! command -v autoreconf >/dev/null 2>&1; then
        SPRAL_WHY="autotools missing (autoconf, automake, libtool)"
    elif [ "$OS" = mac ]; then
        if ! command -v brew >/dev/null 2>&1; then
            SPRAL_WHY="Homebrew not found (needed for metis, hwloc, openblas)"
        else
            for pkg in metis hwloc openblas; do
                if ! brew --prefix "$pkg" >/dev/null 2>&1 \
                   || [ ! -d "$(brew --prefix "$pkg")/lib" ]; then
                    SPRAL_WHY="${SPRAL_WHY:+$SPRAL_WHY; }brew install $pkg"
                fi
            done
            if [ -z "$SPRAL_WHY" ]; then
                METIS_P="$(brew --prefix metis)"; HWLOC_P="$(brew --prefix hwloc)"
                OPENBLAS_P="$(brew --prefix openblas)"
                BLAS_LFLAGS="-L$OPENBLAS_P/lib -lopenblas"
                METIS_LFLAGS="-L$METIS_P/lib -lmetis"; METIS_INC="$METIS_P/include"
                HWLOC_LFLAGS="-L$HWLOC_P/lib -lhwloc"
            fi
        fi
    else
        if [ ! -f /usr/include/metis.h ] && [ ! -f /usr/local/include/metis.h ]; then
            SPRAL_WHY="metis headers missing (apt: libmetis-dev)"
        elif ! ls /usr/lib*/libhwloc.so* /usr/lib/*/libhwloc.so* >/dev/null 2>&1; then
            SPRAL_WHY="hwloc missing (apt: libhwloc-dev)"
        else
            if ls /usr/lib*/libopenblas.so* /usr/lib/*/libopenblas.so* >/dev/null 2>&1; then
                BLAS_LFLAGS="-lopenblas"
            else
                BLAS_LFLAGS="-lblas -llapack"
            fi
            METIS_LFLAGS="-lmetis"
            METIS_INC="$([ -f /usr/include/metis.h ] && echo /usr/include || echo /usr/local/include)"
            HWLOC_LFLAGS="-lhwloc"
        fi
    fi
    if [ -n "$SPRAL_WHY" ]; then
        echo
        echo "note: SPRAL will be skipped: $SPRAL_WHY"
        echo "      (re-run after installing the above to add it)"
        WANT_SPRAL=0
    fi
fi

# Which compiler builds IPOPT: the GCC suite when SPRAL is in the build
# (shared C++ runtime), otherwise whatever configure finds.
if [ "$WANT_SPRAL" = "1" ]; then
    export CC="$GCC_CC" CXX="$GCC_CXX" F77="$GCC_FC" FC="$GCC_FC"
    # GCC's speculative devirtualization emits vtable references to
    # dependency-detector classes IPOPT declares but never compiles (Ma28);
    # the link then fails on a symbol nothing uses.
    export CXXFLAGS="${CXXFLAGS:--O2} -fno-devirtualize-speculatively"
fi
if [ "$WANT_MUMPS" = "1" ] && [ -z "$FC_ANY" ] && [ -z "${FC:-}" ]; then
    echo "error: MUMPS needs gfortran, which was not found" >&2
    echo "       (macOS: brew install gcc; Debian/Ubuntu: apt install gfortran)" >&2
    exit 1
fi

mkdir -p "$IP"
cd "$IP"

# --- sources ---------------------------------------------------------------
# Clone only what is missing, so the script can be re-run after a failed
# build without wiping work already done.
clone() {   # url dir [branch]
    if [ ! -d "$2" ]; then
        if [ -n "${3:-}" ]; then git clone --depth 1 --branch "$3" "$1" "$2"
        else git clone --depth 1 "$1" "$2"; fi
    fi
}
clone https://github.com/coin-or/Ipopt.git Ipopt stable/3.14
clone https://github.com/coin-or-tools/ThirdParty-ASL.git ThirdParty-ASL
[ "$WANT_MUMPS" = "1" ] && clone https://github.com/coin-or-tools/ThirdParty-Mumps.git ThirdParty-Mumps
[ -n "$MA27_SRC" ]      && clone https://github.com/coin-or-tools/ThirdParty-HSL.git ThirdParty-HSL
[ "$WANT_SPRAL" = "1" ] && clone https://github.com/ralna/spral.git spral v2023.03.29

# --- ASL -------------------------------------------------------------------
if [ ! -f "$IP/ASL_build/lib/libcoinasl.$SO" ] && [ ! -f "$IP/ASL_build/lib/libcoinasl.a" ]; then
    say "building ASL"
    cd "$IP/ThirdParty-ASL"
    ./get.ASL
    ./configure --prefix="$IP/ASL_build"
    # separate statements, not `make && make install`: set -e ignores a
    # failure inside an && list, and the script would sail on without it
    make -j"$JOBS"
    make install
fi
ASL_CONFIGURE=(
    --with-asl-cflags="-I$IP/ASL_build/include/coin-or/asl"
    --with-asl-lflags="-L$IP/ASL_build/lib -lcoinasl"
)

# --- MUMPS -----------------------------------------------------------------
MUMPS_CONFIGURE=()
if [ "$WANT_MUMPS" = "1" ]; then
    if [ ! -f "$IP/MUMPS_build/lib/libcoinmumps.$SO" ] && [ ! -f "$IP/MUMPS_build/lib/libcoinmumps.a" ]; then
        say "building MUMPS"
        cd "$IP/ThirdParty-Mumps"
        ./get.Mumps
        ./configure --prefix="$IP/MUMPS_build"
        make -j"$JOBS"
        make install
    fi
    MUMPS_CONFIGURE=(
        --with-mumps-cflags="-I$IP/MUMPS_build/include/coin-or/mumps"
        --with-mumps-lflags="-L$IP/MUMPS_build/lib -lcoinmumps"
    )
fi

# --- MA27 ------------------------------------------------------------------
HSL_CONFIGURE=()
if [ -n "$MA27_SRC" ]; then
    if [ ! -f "$IP/HSL_build/lib/libcoinhsl.$SO" ] && [ ! -f "$IP/HSL_build/lib/libcoinhsl.a" ]; then
        say "building HSL (MA27) from $MA27_SRC"
        # ThirdParty-HSL expects the sources under coinhsl/, with MA27's
        # `src` directory renamed to `ma27`.
        cd "$IP/ThirdParty-HSL"
        rm -rf coinhsl && mkdir -p coinhsl
        cp -a "$MA27_SRC/." coinhsl/
        [ -d coinhsl/src ] && mv coinhsl/src coinhsl/ma27
        ./configure --prefix="$IP/HSL_build"
        make -j"$JOBS"
        make install
    fi
    HSL_CONFIGURE=(
        --with-hsl-cflags="-I$IP/HSL_build/include/coin-or/hsl"
        --with-hsl-lflags="-L$IP/HSL_build/lib -lcoinhsl"
    )
elif [ -f "$IP/HSL_build/lib/libcoinhsl.$SO" ] || [ -f "$IP/HSL_build/lib/libcoinhsl.a" ]; then
    # a root that already carries MA27 keeps it on re-runs
    HSL_CONFIGURE=(
        --with-hsl-cflags="-I$IP/HSL_build/include/coin-or/hsl"
        --with-hsl-lflags="-L$IP/HSL_build/lib -lcoinhsl"
    )
fi

# --- SPRAL -----------------------------------------------------------------
SPRAL_CONFIGURE=()
if [ "$WANT_SPRAL" = "1" ] && [ ! -f "$IP/SPRAL_build/lib/libspral.a" ]; then
    say "building SPRAL"
    # serial make: SPRAL's Makefile does not order its Fortran module
    # dependencies for parallel builds (-j races on spral_*.mod). A failure
    # here costs SPRAL, not the install -- say so and build without it.
    if (cd "$IP/spral" \
        && { [ -f configure ] || ./autogen.sh; } \
        && ./configure --prefix="$IP/SPRAL_build" \
            --with-blas="$BLAS_LFLAGS" --with-lapack="$BLAS_LFLAGS" \
            --with-metis="$METIS_LFLAGS" --with-metis-inc-dir="$METIS_INC" \
        && make \
        && make install); then
        :
    else
        echo
        echo "note: SPRAL failed to compile (see the output above); building"
        echo "      IPOPT without it. Fix the cause and re-run to add it."
        WANT_SPRAL=0
    fi
fi
if [ "$WANT_SPRAL" = "1" ]; then
    GOMP_DIR="$(dirname "$("$GCC_CC" -print-file-name=libgomp.$SO)")"
    GFORT_DIR="$(dirname "$("$GCC_FC" -print-file-name=libgfortran.$SO)")"
    SPRAL_CONFIGURE=(
        --with-spral-cflags="-I$IP/SPRAL_build/include"
        --with-spral-lflags="-L$IP/SPRAL_build/lib -lspral $BLAS_LFLAGS $METIS_LFLAGS $HWLOC_LFLAGS -L$GOMP_DIR -lgomp -L$GFORT_DIR -lgfortran"
    )
fi

# --- IPOPT -----------------------------------------------------------------
build_ipopt() {   # dir  (--enable-static|--enable-shared ...)  extra configure args...
    local dir="$1"; shift
    rm -rf "$dir" && mkdir -p "$dir" && cd "$dir"
    "$IP/Ipopt/configure" --disable-java "${ASL_CONFIGURE[@]}" "$@"
    make -j"$JOBS"
    make install
}

mkdir -p "$IP/build"
if [ "$WANT_SPRAL" = "1" ]; then
    # product 1: the shared library for cyipopt, SPRAL-free (see header)
    say "building IPOPT shared library (cyipopt): MUMPS${HSL_CONFIGURE:+ + MA27}"
    build_ipopt "$IP/Ipopt-build-lib" --prefix="$IP/build" \
        --enable-shared --disable-static \
        "${MUMPS_CONFIGURE[@]}" "${HSL_CONFIGURE[@]}"
    # product 2: the static executable with every solver, staged then copied
    # so the install's pkgconfig/lib stay the shared product's
    say "building IPOPT executable: MUMPS + SPRAL${HSL_CONFIGURE:+ + MA27}"
    build_ipopt "$IP/Ipopt-build-exe" --prefix="$IP/build-exe" \
        --enable-static --disable-shared \
        "${MUMPS_CONFIGURE[@]}" "${HSL_CONFIGURE[@]}" "${SPRAL_CONFIGURE[@]}"
    cp "$IP/build-exe/bin/ipopt" "$IP/build/bin/ipopt"
else
    say "building IPOPT: $([ "$WANT_MUMPS" = 1 ] && echo MUMPS)${HSL_CONFIGURE:+ + MA27}"
    build_ipopt "$IP/Ipopt-build-lib" --prefix="$IP/build" \
        "${MUMPS_CONFIGURE[@]}" "${HSL_CONFIGURE[@]}"
fi

# --- check -----------------------------------------------------------------
EXE="$IP/build/bin/ipopt"
if env -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH "$EXE" --version >/dev/null 2>&1; then
    NEEDS_LIBPATH=0
else
    NEEDS_LIBPATH=1
fi
# which solvers the executable actually carries: minimize (x-3)^2 (an AMPL
# .nl as pyomo writes it) under each; IPOPT writes no .sol when it rejects
# the linear_solver option
PROBE="$IP/build/.probe.nl"
cat > "$PROBE" <<'EOF'
g3 1 1 0
 1 0 1 0 0
 0 1 0 0 0 0
 0 0
 0 1 0
 0 0 0 1
 0 0 0 0 0
 0 1
 0 0
 0 0 0 0 0
O0 0
o5
o0
v0
n-3.0
n2
x1
0 1.0
r
b
0 0 10
k0
G0 1
0 0
EOF
HAVE=""
for s in ma27 mumps spral; do
    if OMP_CANCELLATION=TRUE OMP_PROC_BIND=TRUE "$EXE" "$PROBE" -AMPL "linear_solver=$s" print_level=0 2>&1 \
       | grep -qi "invalid\|not available\|unknown\|not been compiled\|not supported"; then
        :
    elif [ -f "${PROBE%.nl}.sol" ]; then
        HAVE="$HAVE $s"; rm -f "${PROBE%.nl}.sol"
    fi
done
rm -f "$PROBE"

echo
echo "--------------------------------------------------------------------------"
echo "Build complete: $EXE"
echo "Linear solvers in the executable:${HAVE:- (none detected -- see above)}"
if [ "$WANT_SPRAL" = "1" ]; then
    echo "Shared library (cyipopt, in-process): MUMPS${HSL_CONFIGURE:+ + MA27} -- SPRAL is"
    echo "executable-only (see the header of this script for why)."
fi
echo
if [ "$NEEDS_LIBPATH" = "1" ]; then
    cat <<EOF
This build needs its libraries on the loader path -- it was checked just now
and would not start without them. Add to your shell profile:

export $LIBVAR="$IP/build/lib:$IP/ASL_build/lib:$IP/MUMPS_build/lib:$IP/HSL_build/lib:\$$LIBVAR"

EOF
else
    echo "Checked: the binary starts with no $LIBVAR set."
    echo
fi
cat <<EOF
LCsolver records this build and uses it without any PATH change. With no
MA27 it defaults to SPRAL; with MA27 to MA27; every solver in the build is
selectable per solve with linear_solver=. Verify with:

    lcsolver-check-solvers

To add MA27 later (free for academic use, fetched by hand):

    $0 $ROOT --add-ma27 /path/to/ma27-1.0.0

To run 'ipopt' by hand in a terminal:

export PATH="$IP/build/bin:\$PATH"
--------------------------------------------------------------------------
EOF
