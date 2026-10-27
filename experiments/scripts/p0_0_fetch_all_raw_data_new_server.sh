#!/bin/bash
# P0.0  Re-create experiments/data/raw on a new server (migration from LUMI, 2026-10-27).
# Downloads every raw input used by P0-P7 from its original source, then checks the files
# against experiments/data_manifest_sha256.txt (written on LUMI before the migration).
# About 6.5 GB. The DIAMOND binary is fetched by p1_0b; d4 is built by tools/build_exact_solvers.sh.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$EXP_DIR/data/raw"
mkdir -p "$RAW"

get() {  # dest url
    local dest="$1" url="$2"
    mkdir -p "$(dirname "$dest")"
    if [[ -s "$dest" ]]; then echo "exists: ${dest#$EXP_DIR/}"; return; fi
    echo "downloading ${dest#$EXP_DIR/}"
    curl -fL --retry 3 -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

# ontologies (GO 2021-11-16, UBERON 2026-06-19, ChEBI 252) and Swiss-Prot 2021_04 + DIAMOND v2.2.6
bash "$EXP_DIR/scripts/p0_1_download_paper_ontology_releases.sh"
bash "$EXP_DIR/scripts/p1_0b_download_swissprot_2021_04_and_diamond.sh"

# GO 2024-01-17 and Swiss-Prot 2024_01 (P5 / P7 time-split labels)
get "$RAW/go-basic_2024-01-17.obo" http://release.geneontology.org/2024-01-17/ontology/go-basic.obo
U24="$RAW/uniprot_2024_01"
get "$U24/uniprot_sprot-only2024_01.tar.gz" \
    https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2024_01/knowledgebase/uniprot_sprot-only2024_01.tar.gz
[[ -s "$U24/uniprot_sprot.dat.gz" ]] || tar -xzf "$U24/uniprot_sprot-only2024_01.tar.gz" -C "$U24" uniprot_sprot.dat.gz uniprot_sprot.fasta.gz

# DeepGOPlus 1.0.6 release data (model, train/test data, predictions, test_diamond.res)
DGP="$RAW/deepgoplus_1.0.6"
get "$DGP/data-1.0.6.tar.gz" https://bio2vec.net/data/deepgo/data-1.0.6.tar.gz
[[ -s "$DGP/data/predictions.pkl" ]] || tar -xzf "$DGP/data-1.0.6.tar.gz" -C "$DGP"

# verify against the LUMI manifest (raw files only; derived files come from the data bundle)
cd "$EXP_DIR/data"
grep '  raw/' "$EXP_DIR/data_manifest_sha256.txt" | while read -r sum path; do
    if [[ ! -f "$path" ]]; then echo "MISSING  $path"
    elif [[ "$(sha256sum "$path" | cut -d' ' -f1)" != "$sum" ]]; then echo "DIFFERS  $path"
    fi
done | tee "$EXP_DIR/results/p0_0_raw_data_check_$(hostname -s).txt"
echo "raw data check done: lines above (if any) list missing or differing files"
