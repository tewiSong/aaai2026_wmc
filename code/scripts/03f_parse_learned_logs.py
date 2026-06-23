"""Parse actual learned-WMC SLURM logs into structured experiment results.

The A-NeSI and NeSyDM carry-chain runs are external systems, so this script records the
numbers from their completed SLURM logs instead of retyping them in the paper. It fails if
the expected lines are absent; missing values are not inferred.
"""

from __future__ import annotations

import json
import os
import re
from glob import glob

import lib

SLURM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "slurm")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _latest(pattern: str) -> str:
    paths = sorted(glob(os.path.join(SLURM_DIR, pattern)), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError(f"no SLURM log matched {pattern}")
    return paths[-1]


def parse_anesi(path: str) -> dict:
    text = _read(path)
    pat = re.compile(r"^A-NeSI N=(\d+) final: Val accuracy: ([0-9.]+)$", re.MULTILINE)
    rows = {int(n): float(acc) for n, acc in pat.findall(text)}
    expected = {1, 2, 3, 4}
    missing = sorted(expected - set(rows))
    if missing:
        raise RuntimeError(f"A-NeSI log {path} is missing final rows for N={missing}")
    return {
        "log": os.path.relpath(path, os.path.join(SLURM_DIR, "..", "..")),
        "sum_accuracy": {str(n): rows[n] for n in sorted(expected)},
    }


def parse_nesydm(path: str) -> dict:
    text = _read(path)
    pat = re.compile(r"^NeSyDM N=(\d+) best answer-acc: y_fTM=([0-9.]+)", re.MULTILINE)
    rows = {int(n): float(acc) for n, acc in pat.findall(text)}
    expected = {1, 4}
    missing = sorted(expected - set(rows))
    if missing:
        raise RuntimeError(f"NeSyDM log {path} is missing endpoint rows for N={missing}")
    return {
        "log": os.path.relpath(path, os.path.join(SLURM_DIR, "..", "..")),
        "answer_accuracy": {str(n): rows[n] for n in sorted(expected)},
    }


def main() -> None:
    lib.ensure_results_dir()
    anesi_log = _latest("nesy_gpu_anesi.*.out")
    nesydm_log = _latest("nesy_gpu_nesydm.*.out")
    out = {
        "anesi": parse_anesi(anesi_log),
        "nesydm": parse_nesydm(nesydm_log),
    }
    path = os.path.join(lib.RESULTS_DIR, "exp3f_learned_logs.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
