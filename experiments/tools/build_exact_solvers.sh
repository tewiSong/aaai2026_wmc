#!/bin/bash
# Build exact model counters used by P1 inside the project container.
# Sources are cloned into tools/src at pinned commits; binaries go to tools/bin.
#   d4 (crillab/d4, the competition version): decision-DNNF compiler (-dDNNF -out=...)
set -euo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$TOOLS/src"
BIN="$TOOLS/bin"
mkdir -p "$SRC" "$BIN"

D4_COMMIT=333370cc1e843dd0749c1efe88516e72b5239174
if [[ ! -d "$SRC/d4" ]]; then
    git clone -q https://github.com/crillab/d4.git "$SRC/d4"
fi
cd "$SRC/d4"
git checkout -q -- . && git checkout -q "$D4_COMMIT"
# Patch: DAG<T>::computeNbModels is declared virtual but never defined; old GCC let the
# vtable reference slide, current GCC/ld do not. Give the base version a body.
sed -i 's|^  virtual T computeNbModels();|  virtual T computeNbModels() { return T(0); }|' DAG/DAG.hh
grep -q 'virtual T computeNbModels() { return T(0); }' DAG/DAG.hh
P="$CONDA_PREFIX"
make -j8 s CXX="g++ -std=c++17" \
    CFLAGS="-I$P/include" \
    LFLAGS="-L$P/lib -Wl,-rpath,$P/lib -lz -lgmpxx -lgmp patoh/libpatoh.a"
cp d4 "$BIN/d4"
echo "built $BIN/d4 at commit $D4_COMMIT"
