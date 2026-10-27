#!/bin/bash
# P0.1  Download the exact ontology releases used by the AAAI manuscript and record
# checksums and version headers, so every later number is tied to a fixed input.
#   GO     release 2021-11-16 (go-basic.obo)
#   UBERON versionIRI .../releases/2026-06-19/  (published as GitHub tag v2026-06-23)
#   ChEBI  data-version 252
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$EXP_DIR/data/raw"
OUT="$EXP_DIR/results/p0_1_ontology_release_manifest.tsv"
mkdir -p "$RAW"

fetch() {  # name url
    local name="$1" url="$2"
    if [[ -s "$RAW/$name" ]]; then
        echo "exists: $name"
    else
        echo "downloading $name"
        curl -fL --retry 3 -o "$RAW/$name.part" "$url"
        mv "$RAW/$name.part" "$RAW/$name"
    fi
}

fetch go-basic_2021-11-16.obo     http://release.geneontology.org/2021-11-16/ontology/go-basic.obo
fetch uberon-basic_2026-06-19.obo https://github.com/obophenotype/uberon/releases/download/v2026-06-23/uberon-basic.obo
fetch uberon-basic_2026-06-19.owl https://github.com/obophenotype/uberon/releases/download/v2026-06-23/uberon-basic.owl
fetch chebi_rel252.obo.gz         https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel252/ontology/chebi.obo.gz
[[ -s "$RAW/chebi_rel252.obo" ]] || gunzip -k "$RAW/chebi_rel252.obo.gz"

{
    printf "file\tbytes\tsha256\tversion_header\n"
    for f in go-basic_2021-11-16.obo uberon-basic_2026-06-19.obo uberon-basic_2026-06-19.owl chebi_rel252.obo; do
        case "$f" in
            *.obo) ver="$(head -n 30 "$RAW/$f" | grep -m1 -E '^data-version:' || true)" ;;
            *.owl) ver="$(head -c 20000 "$RAW/$f" | grep -m1 -o 'versionIRI rdf:resource="[^"]*"' || true)" ;;
        esac
        printf "%s\t%s\t%s\t%s\n" "$f" "$(stat -c %s "$RAW/$f")" "$(sha256sum "$RAW/$f" | cut -d' ' -f1)" "$ver"
    done
} > "$OUT"

cat "$OUT"
# Expected headers: GO releases/2021-11-16, UBERON .../2026-06-19/..., ChEBI 252.
grep -q "2021-11-16" "$OUT" && grep -q "2026-06-19" "$OUT" && grep -q "data-version: 252" "$OUT" \
    && echo "P0.1 OK: all three releases match the manuscript" \
    || { echo "P0.1 MISMATCH: a version header differs from the manuscript"; exit 2; }
