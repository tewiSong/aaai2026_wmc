"""Exp 0: build the GO true-path theories and report dataset statistics.

Builds one TruePathTheory per namespace (cc / mf / bp) from go-basic.obo, caches them,
and writes a stats table: term count, edge count, connected components, and the fraction
of multi-parent (reconvergent) terms. These are the inputs to every later experiment.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import networkx as nx

import lib


def main() -> None:
    lib.ensure_results_dir()
    theories = lib.load_theories()

    rows = []
    for ns in lib.NAMESPACES:
        th = theories[ns]
        g = th.constraint_graph()
        indeg = Counter()
        for c, _p in th.nf1:
            indeg[c] += 1
        multi_parent = sum(1 for a in th.atoms if indeg[a] > 1)
        row = dict(
            namespace=ns,
            n_terms=len(th.atoms),
            n_edges=len(th.nf1),
            n_components=nx.number_connected_components(g),
            multi_parent=multi_parent,
            multi_parent_frac=round(multi_parent / len(th.atoms), 4),
            max_in_degree=max(indeg.values()) if indeg else 0,
        )
        rows.append(row)
        print(f"{ns}: terms={row['n_terms']:6d} edges={row['n_edges']:6d} "
              f"components={row['n_components']:4d} "
              f"multi-parent={multi_parent:6d} ({100*row['multi_parent_frac']:.0f}%) "
              f"max_in_degree={row['max_in_degree']}", flush=True)

    out = os.path.join(lib.RESULTS_DIR, "exp0_dataset_stats.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
