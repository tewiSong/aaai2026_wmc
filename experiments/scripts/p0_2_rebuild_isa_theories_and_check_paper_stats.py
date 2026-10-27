"""P0.2  Rebuild the is-a implication theories and compare with the manuscript.

Purpose: before any baseline is run, confirm that our encoding (live terms, is-a edges
only, one clause y_child => y_parent) reproduces the manuscript's term counts and
structure statistics. Any later comparison is meaningless if the theories differ.

Outputs
  data/theories/<name>.npz, data/theories/<name>.cnf
  results/p0_2_ontology_structure_vs_paper.tsv
"""
from __future__ import annotations

from collections import deque

import numpy as np

from wmc_common import (PAPER_STRUCTURE, RAW_DIR, RESULTS_DIR, THEORY_DIR, build_isa_theory,
                        parse_obo_terms, save_theory, write_dimacs)

GO_NAMESPACES = {"go_cc": "cellular_component", "go_mf": "molecular_function",
                 "go_bp": "biological_process"}


def structure_stats(n: int, edges: np.ndarray) -> dict:
    parents = np.bincount(edges[:, 0], minlength=n) if len(edges) else np.zeros(n, int)
    children = np.bincount(edges[:, 1], minlength=n) if len(edges) else np.zeros(n, int)

    # acyclicity (Kahn) on child -> parent edges
    out_adj = [[] for _ in range(n)]
    for c, p in edges:
        out_adj[c].append(p)
    indeg = children.copy()
    queue = deque(np.flatnonzero(indeg == 0).tolist())
    seen = 0
    while queue:
        v = queue.popleft()
        seen += 1
        for p in out_adj[v]:
            indeg[p] -= 1
            if indeg[p] == 0:
                queue.append(p)

    # weakly connected components (union-find)
    root = list(range(n))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for c, p in edges:
        a, b = find(int(c)), find(int(p))
        if a != b:
            root[a] = b
    n_comp = len({find(v) for v in range(n)})

    return dict(
        terms=n, edges=len(edges),
        pct_mp=round(100.0 * float(np.mean(parents >= 2)), 2),
        max_parents=int(parents.max()), max_children=int(children.max()),
        max_undirected_deg=int((parents + children).max()),
        roots=int(np.sum(parents == 0)), components=n_comp, is_dag=(seen == n),
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    THEORY_DIR.mkdir(parents=True, exist_ok=True)
    theories = {}

    go = parse_obo_terms(RAW_DIR / "go-basic_2021-11-16.obo")
    for name, ns in GO_NAMESPACES.items():
        theories[name] = build_isa_theory(go, lambda t, ns=ns: t["namespace"] == ns)

    ub = parse_obo_terms(RAW_DIR / "uberon-basic_2026-06-19.obo")
    theories["uberon"] = build_isa_theory(ub, lambda t: t["id"].startswith("UBERON:"))
    theories["uberon_allterms"] = build_isa_theory(ub, lambda t: True)

    ch = parse_obo_terms(RAW_DIR / "chebi_rel252.obo")
    theories["chebi"] = build_isa_theory(ch, lambda t: True)

    cols = ["name", "terms", "paper_terms", "edges", "dropped_isa", "pct_mp", "paper_pct_mp",
            "max_parents", "max_children", "max_undirected_deg", "paper_max_deg",
            "roots", "components", "is_dag", "terms_match"]
    rows = []
    for name, (ids, edges, dropped) in theories.items():
        save_theory(name, ids, edges)
        write_dimacs(THEORY_DIR / f"{name}.cnf", len(ids), edges, comment=f"is-a theory {name}")
        st = structure_stats(len(ids), edges)
        paper = PAPER_STRUCTURE.get(name.replace("_allterms", ""), {})
        row = dict(name=name, dropped_isa=dropped, paper_terms=paper.get("terms", ""),
                   paper_pct_mp=paper.get("pct_mp", ""), paper_max_deg=paper.get("max_deg", ""),
                   terms_match=(st["terms"] == paper.get("terms")), **st)
        rows.append(row)
        print({k: row[k] for k in cols}, flush=True)

    out = RESULTS_DIR / "p0_2_ontology_structure_vs_paper.tsv"
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")
    print(f"\nwritten {out}")
    print("Which degree column matches the paper's 'Max deg.' is decided from this table;"
          " a term-count mismatch must be explained before P1.")


if __name__ == "__main__":
    main()
