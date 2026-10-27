"""Shared helpers: OBO parsing, is-a implication theories, and their I/O.

A theory is stored as data/theories/<name>.npz with
    ids    : term identifiers, index i <-> Boolean variable y_{i+1}
    edges  : int32 array (E, 2) of (child, parent) index pairs, one clause y_c => y_p each
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
THEORY_DIR = DATA_DIR / "theories"
ORDER_DIR = DATA_DIR / "orders"
RESULTS_DIR = EXP_DIR / "results"

ONTOLOGY_NAMES = ["go_cc", "go_mf", "go_bp", "uberon", "chebi"]

# Structure table of the AAAI manuscript (technical supplement, "Unrestricted modular
# pre-screen"): terms, multi-parent %, max degree, min-fill width of the whole theory.
PAPER_STRUCTURE = {
    "go_cc": dict(terms=4041, pct_mp=13.0, max_deg=4, minfill_width=12),
    "go_mf": dict(terms=10131, pct_mp=20.0, max_deg=7, minfill_width=37),
    "go_bp": dict(terms=25950, pct_mp=53.0, max_deg=9, minfill_width=269),
    "uberon": dict(terms=14977, pct_mp=24.0, max_deg=7, minfill_width=46),
    "chebi": dict(terms=205592, pct_mp=24.7, max_deg=14, minfill_width=738),
}


def parse_obo_terms(path: Path) -> dict[str, dict]:
    """Return {id: {namespace, obsolete, is_a: [ids]}} for every [Term] stanza."""
    terms: dict[str, dict] = {}
    cur = None
    in_term = False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("["):
                if cur is not None:
                    terms[cur["id"]] = cur
                cur = None
                in_term = line.strip() == "[Term]"
                if in_term:
                    cur = {"id": None, "namespace": None, "obsolete": False, "is_a": []}
                continue
            if not in_term or cur is None or ":" not in line:
                continue
            key, _, val = line.partition(":")
            val = val.strip()
            if key == "id":
                cur["id"] = val.split()[0]
            elif key == "namespace":
                cur["namespace"] = val
            elif key == "is_obsolete":
                cur["obsolete"] = val.lower().startswith("true")
            elif key == "is_a":
                cur["is_a"].append(val.split()[0])
    if cur is not None:
        terms[cur["id"]] = cur
    terms.pop(None, None)
    return terms


def build_isa_theory(terms: dict[str, dict], keep) -> tuple[list[str], np.ndarray, int]:
    """Keep live terms satisfying keep(term); return ids, (child, parent) edges and the
    number of is-a links dropped because the parent is outside the kept set."""
    ids = sorted(t["id"] for t in terms.values() if not t["obsolete"] and keep(t))
    index = {tid: i for i, tid in enumerate(ids)}
    edges, dropped = set(), 0
    for tid in ids:
        c = index[tid]
        for parent in terms[tid]["is_a"]:
            p = index.get(parent)
            if p is None:
                dropped += 1
            elif p != c:
                edges.add((c, p))
    arr = np.array(sorted(edges), dtype=np.int32).reshape(-1, 2)
    return ids, arr, dropped


def save_theory(name: str, ids: list[str], edges: np.ndarray) -> Path:
    THEORY_DIR.mkdir(parents=True, exist_ok=True)
    path = THEORY_DIR / f"{name}.npz"
    np.savez_compressed(path, ids=np.array(ids), edges=edges)
    return path


def load_theory(name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(THEORY_DIR / f"{name}.npz") as z:
        return z["ids"], z["edges"]


def write_dimacs(path: Path, n_vars: int, edges: np.ndarray, comment: str = "") -> None:
    """Unweighted CNF: clause (-y_c v y_p) per edge, variables 1-indexed."""
    with open(path, "w") as fh:
        if comment:
            fh.write(f"c {comment}\n")
        fh.write(f"p cnf {n_vars} {len(edges)}\n")
        for c, p in edges:
            fh.write(f"-{c + 1} {p + 1} 0\n")


def undirected_adjacency(n: int, edges: np.ndarray) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for c, p in edges:
        adj[c].add(int(p))
        adj[p].add(int(c))
    return adj
