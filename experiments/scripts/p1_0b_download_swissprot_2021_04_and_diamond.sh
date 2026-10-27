#!/bin/bash
# P1.0b  Inputs for realistic predictor scores: Swiss-Prot release 2021_04 (the UniProt
# release closest to the GO 2021-11-16 snapshot used for the theories) and a pinned
# DIAMOND binary for the DiamondScore homology baseline (standard CAFA baseline).
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$EXP_DIR/data/raw/uniprot_2021_04"
BIN="$EXP_DIR/tools/bin"
mkdir -p "$RAW" "$BIN"

TAR="$RAW/uniprot_sprot-only2021_04.tar.gz"
if [[ ! -s "$TAR" ]]; then
    curl -fL --retry 3 -o "$TAR.part" \
        https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2021_04/knowledgebase/uniprot_sprot-only2021_04.tar.gz
    mv "$TAR.part" "$TAR"
fi
tar -tzf "$TAR" | head -20
tar -xzf "$TAR" -C "$RAW" uniprot_sprot.dat.gz uniprot_sprot.fasta.gz

DIAMOND_VERSION=v2.2.6
if [[ ! -x "$BIN/diamond" ]]; then
    curl -fL --retry 3 -o "$BIN/diamond.tar.gz" \
        "https://github.com/bbuchfink/diamond/releases/download/$DIAMOND_VERSION/diamond-linux64.tar.gz"
    tar -xzf "$BIN/diamond.tar.gz" -C "$BIN" && rm "$BIN/diamond.tar.gz"
fi
"$BIN/diamond" version
sha256sum "$TAR" "$RAW"/uniprot_sprot.*.gz "$BIN/diamond" | tee "$EXP_DIR/results/p1_0b_inputs_sha256.txt"
