"""True-path theory construction from the Gene Ontology.

The true-path theory T over a target vocabulary Y (the GO terms of one namespace) is

    T = AND_{(c,p) in E} (y_c -> y_p)                      # is-a / part-of edges
        AND_{A1&A2 -> B in NF2} (y_A1 & y_A2 -> y_B)       # conjunction definitions
        AND existential definitions (NF3/NF4)              # see normalize.py

This module builds the hierarchical backbone E directly from the OBO release
(`go-basic.obo`): for every non-obsolete term, its `is_a` parents and its
`part_of` (relationship) parents become true-path edges child -> parent, which is
exactly the GO "true-path rule". Terms are partitioned by namespace
(cellular_component / molecular_function / biological_process).

The NF2/NF3/NF4 enrichment is produced separately by `normalize.py` from `go.owl`
and merged through `TruePathTheory.add_nf2` / `add_existential`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import networkx as nx

NAMESPACE_SHORT = {
    "cellular_component": "cc",
    "molecular_function": "mf",
    "biological_process": "bp",
}


@dataclass
class OboTerm:
    go_id: str
    name: str
    namespace: str
    is_a: List[str] = field(default_factory=list)
    part_of: List[str] = field(default_factory=list)
    is_obsolete: bool = False
    alt_ids: List[str] = field(default_factory=list)
    replaced_by: List[str] = field(default_factory=list)


def parse_obo(path: str) -> Dict[str, OboTerm]:
    """Parse the [Term] stanzas of an OBO file.

    Returns a dict from canonical GO id to OboTerm. Only the fields needed to build
    the true-path backbone are extracted. The parser is exact for the GO OBO format:
    it tracks stanza boundaries, strips trailing ' ! label' comments, and reads
    `relationship: part_of TARGET` lines.
    """
    terms: Dict[str, OboTerm] = {}
    cur: OboTerm | None = None
    in_term = False

    def flush(t: OboTerm | None) -> None:
        if t is not None and t.go_id:
            terms[t.go_id] = t

    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("[") and line.endswith("]"):
                # New stanza: commit the previous term, start fresh.
                flush(cur)
                cur = None
                in_term = line == "[Term]"
                if in_term:
                    cur = OboTerm(go_id="", name="", namespace="")
                continue
            if not in_term or cur is None:
                continue
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            # Drop trailing dbxref/comment ' ! ...'
            if " ! " in value:
                value = value.split(" ! ", 1)[0].strip()
            if key == "id":
                cur.go_id = value
            elif key == "name":
                cur.name = value
            elif key == "namespace":
                cur.namespace = value
            elif key == "is_a":
                cur.is_a.append(value.split()[0])
            elif key == "relationship":
                parts = value.split()
                if len(parts) >= 2 and parts[0] == "part_of":
                    cur.part_of.append(parts[1])
            elif key == "is_obsolete":
                cur.is_obsolete = value.lower() == "true"
            elif key == "alt_id":
                cur.alt_ids.append(value)
            elif key == "replaced_by":
                cur.replaced_by.append(value)
    flush(cur)
    return terms


@dataclass
class TruePathTheory:
    """A propositional true-path theory over one GO namespace.

    Atoms are GO ids. Clauses are stored in three groups matching the EL++ normal
    forms; `constraint_graph()` turns them into the moralized constraint graph
    (one clique per axiom factor) whose treewidth governs exact WMC cost.
    """

    namespace: str
    atoms: List[str]
    nf1: List[Tuple[str, str]] = field(default_factory=list)  # (child, parent): child -> parent
    nf2: List[Tuple[str, str, str]] = field(default_factory=list)  # (a, b, e): a & b -> e
    nf3: List[Tuple[str, str]] = field(default_factory=list)  # (a, b): a -> exists r.b  (grounded a -> b)
    nf4: List[Tuple[str, str]] = field(default_factory=list)  # (b, e): exists r.b -> e  (grounded b -> e)

    def __post_init__(self) -> None:
        self._atom_set: Set[str] = set(self.atoms)

    # -- construction helpers -------------------------------------------------
    def add_nf2(self, triples: List[Tuple[str, str, str]]) -> None:
        for a, b, e in triples:
            if a in self._atom_set and b in self._atom_set and e in self._atom_set:
                self.nf2.append((a, b, e))

    def add_existential(self, nf3: List[Tuple[str, str]], nf4: List[Tuple[str, str]]) -> None:
        for a, b in nf3:
            if a in self._atom_set and b in self._atom_set:
                self.nf3.append((a, b))
        for b, e in nf4:
            if b in self._atom_set and e in self._atom_set:
                self.nf4.append((b, e))

    # -- views ----------------------------------------------------------------
    def clauses(self) -> List[Tuple[int, ...]]:
        """All Horn clauses as tuples of signed atom-indices (1-based, negative = negated).

        index(atom) = position in self.atoms + 1.
        NF1 child->parent      : (-child, parent)
        NF2 a&b->e             : (-a, -b, e)
        NF3 a->exists r.b      : (-a, b)    (propositional true-path grounding)
        NF4 exists r.b->e      : (-b, e)    (propositional true-path grounding)
        """
        idx = {a: i + 1 for i, a in enumerate(self.atoms)}
        cl: List[Tuple[int, ...]] = []
        for c, p in self.nf1:
            cl.append((-idx[c], idx[p]))
        for a, b, e in self.nf2:
            cl.append((-idx[a], -idx[b], idx[e]))
        for a, b in self.nf3:
            cl.append((-idx[a], idx[b]))
        for b, e in self.nf4:
            cl.append((-idx[b], idx[e]))
        return cl

    def implication_edges(self) -> List[Tuple[str, str]]:
        """Directed implication edges atom -> atom that drive the soft closure.

        Each clause (¬x ∨ y) gives an edge x -> y (x is a *sufficient condition* of y).
        NF2 a&b->e contributes the joint sufficient condition {a,b} -> e, returned by
        `nf2` separately; here we return only the unary edges used by soft-OR over
        single parents, plus we keep NF2 for the conjunction-aware update.
        """
        edges = []
        for c, p in self.nf1:
            edges.append((c, p))
        for a, b in self.nf3:
            edges.append((a, b))
        for b, e in self.nf4:
            edges.append((b, e))
        return edges

    def constraint_graph(self) -> nx.Graph:
        """Moralized (undirected) constraint graph: one clique per axiom factor."""
        g = nx.Graph()
        g.add_nodes_from(self.atoms)
        for c, p in self.nf1:
            g.add_edge(c, p)
        for a, b, e in self.nf2:
            g.add_edge(a, b)
            g.add_edge(a, e)
            g.add_edge(b, e)
        for a, b in self.nf3:
            g.add_edge(a, b)
        for b, e in self.nf4:
            g.add_edge(b, e)
        return g


def build_namespace_theories_from_norm(norm_dir: str,
                                       include_existential: bool = False) -> Dict[str, TruePathTheory]:
    """Build per-namespace theories from EL++ normalized axioms (mowl/ELK output).

    Expects `norm_dir/<ns>.json` with keys atoms, nf1 [[c,p]], nf2 [[a,b,e]],
    nf3 [[a,b]] (A -> exists r.B grounded as A,B), nf4 [[b,e]] (exists r.B -> E grounded
    as B,E). The Horn core is gci0 (nf1) + gci1 (nf2); existential factors (nf3/nf4) are
    included only when `include_existential` is set, since over the named vocabulary they
    reintroduce the role partonomy.
    """
    import json
    theories: Dict[str, TruePathTheory] = {}
    for short_ns in NAMESPACE_SHORT.values():
        path = os.path.join(norm_dir, f"{short_ns}.json")
        with open(path) as fh:
            d = json.load(fh)
        th = TruePathTheory(namespace=short_ns, atoms=d["atoms"],
                            nf1=[tuple(x) for x in d.get("nf1", [])],
                            nf2=[tuple(x) for x in d.get("nf2", [])])
        if include_existential:
            th.add_existential([tuple(x) for x in d.get("nf3", [])],
                               [tuple(x) for x in d.get("nf4", [])])
        theories[short_ns] = th
    return theories


def build_namespace_theories(obo_path: str, include_part_of: bool = False) -> Dict[str, TruePathTheory]:
    """Build one TruePathTheory per namespace from `go-basic.obo`.

    Obsolete terms are dropped. An edge is kept only if both endpoints are
    non-obsolete terms of the *same* namespace (GO is-a/part-of edges do not cross
    namespaces; the rare exceptions are regulatory cross-edges absent from the
    is_a/part_of backbone).

    `include_part_of` controls whether the part-of partonomy is added to the
    subsumption backbone. The EL++ normal form treats is-a as the subsumption
    relation (gci0) and part-of as an existential role (gci2); the constraint graph
    whose treewidth governs exact WMC is the subsumption hierarchy, so the default is
    is-a only. Setting it True adds the part-of edges (used for the annotation-style
    true-path rule and as a stress configuration).
    """
    terms = parse_obo(obo_path)
    live = {gid: t for gid, t in terms.items() if not t.is_obsolete}
    ns_of = {gid: t.namespace for gid, t in live.items()}

    theories: Dict[str, TruePathTheory] = {}
    for long_ns, short_ns in NAMESPACE_SHORT.items():
        atoms = sorted(gid for gid, t in live.items() if t.namespace == long_ns)
        atom_set = set(atoms)
        nf1: List[Tuple[str, str]] = []
        for gid in atoms:
            t = live[gid]
            parents = t.is_a + (t.part_of if include_part_of else [])
            for parent in parents:
                if parent in atom_set and ns_of.get(parent) == long_ns:
                    nf1.append((gid, parent))
        theories[short_ns] = TruePathTheory(namespace=short_ns, atoms=atoms, nf1=nf1)
    return theories
