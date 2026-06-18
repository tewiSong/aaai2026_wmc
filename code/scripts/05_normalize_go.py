"""Task A: EL++ normalize go.owl with mowl/ELK into NF1-NF4, partitioned by namespace.

Runs in the dedicated `nesy-mowl` conda env (mowl-borg + JVM). Produces, per namespace,
  <out>/<ns>.json = {atoms, nf1:[[c,p]], nf2:[[a,b,e]], nf3:[[a,b]], nf4:[[b,e]]}
where the EL++ normal forms are
  gci0  C        sub D        -> nf1 (C,D)         clause (~C | D)
  gci1  C1 & C2  sub D        -> nf2 (C1,C2,D)     clause (~C1 | ~C2 | D)
  gci2  C        sub exists R.D -> nf3 (C,D)       (grounded C -> D)
  gci3  exists R.C sub D      -> nf4 (C,D)         (grounded C -> D)
Only axioms all of whose named classes are GO terms of the same namespace are kept.

Usage:
  /ibex/user/songt/conda_envs/nesy-mowl/bin/python 05_normalize_go.py \
      --owl data/go.owl --obo data/go-basic.obo --out data/processed/norm
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import mowl
mowl.init_jvm("16g")

from mowl.ontology.normalize import ELNormalizer  # noqa: E402
from org.semanticweb.owlapi.apibinding import OWLManager  # noqa: E402
from java.io import File  # noqa: E402


OBO_PREFIX = "http://purl.obolibrary.org/obo/"


def iri_to_go(iri: str):
    """Map an OWL class IRI to a GO id, or None if it is not a GO term."""
    s = str(iri)
    if s.startswith(OBO_PREFIX):
        local = s[len(OBO_PREFIX):]
        if local.startswith("GO_"):
            return "GO:" + local[3:]
    return None


def parse_namespaces(obo_path):
    ns = {}
    cur = None
    with open(obo_path) as fh:
        in_term = False
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                in_term = True
                cur = None
                continue
            if line.startswith("["):
                in_term = False
                continue
            if not in_term:
                continue
            if line.startswith("id:"):
                cur = line.split(":", 1)[1].strip()
            elif line.startswith("namespace:") and cur:
                ns[cur] = line.split(":", 1)[1].strip()
    return ns


NS_SHORT = {"cellular_component": "cc", "molecular_function": "mf", "biological_process": "bp"}


def gci_class_name(obj):
    """Return the IRI string of a mowl GCI class field across API variants."""
    for attr in ("name", "iri", "str"):
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            return str(v() if callable(v) else v)
    return str(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owl", required=True)
    ap.add_argument("--obo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ns_of = parse_namespaces(args.obo)
    short_of = {gid: NS_SHORT[ns] for gid, ns in ns_of.items() if ns in NS_SHORT}

    manager = OWLManager.createOWLOntologyManager()
    ont = manager.loadOntologyFromOntologyDocument(File(args.owl))
    print("loaded ontology; normalizing ...", flush=True)
    gcis = ELNormalizer().normalize(ont)
    print("normalization keys:", list(gcis.keys()), flush=True)

    # Inspect one object of each kind for attribute names (debug aid).
    for k in ("gci0", "gci1", "gci2", "gci3"):
        if gcis.get(k):
            o = gcis[k][0]
            print(f"  {k} sample attrs:", [a for a in dir(o) if not a.startswith("_")][:12], flush=True)

    per_ns = defaultdict(lambda: {"atoms": set(), "nf1": [], "nf2": [], "nf3": [], "nf4": []})

    def same_ns(*gids):
        nss = {short_of.get(g) for g in gids}
        return len(nss) == 1 and None not in nss

    def add_atoms(ns, *gids):
        for g in gids:
            per_ns[ns]["atoms"].add(g)

    # gci0: C sub D
    for o in gcis.get("gci0", []):
        c = iri_to_go(gci_class_name(o.subclass)); d = iri_to_go(gci_class_name(o.superclass))
        if c and d and same_ns(c, d):
            ns = short_of[c]; per_ns[ns]["nf1"].append([c, d]); add_atoms(ns, c, d)
    # gci1: C1 & C2 sub D
    for o in gcis.get("gci1", []):
        a = iri_to_go(gci_class_name(o.left_subclass))
        b = iri_to_go(gci_class_name(o.right_subclass))
        d = iri_to_go(gci_class_name(o.superclass))
        if a and b and d and same_ns(a, b, d):
            ns = short_of[a]; per_ns[ns]["nf2"].append([a, b, d]); add_atoms(ns, a, b, d)
    # gci2: C sub exists R.D
    for o in gcis.get("gci2", []):
        c = iri_to_go(gci_class_name(o.subclass)); d = iri_to_go(gci_class_name(o.filler))
        if c and d and same_ns(c, d):
            ns = short_of[c]; per_ns[ns]["nf3"].append([c, d]); add_atoms(ns, c, d)
    # gci3: exists R.C sub D
    for o in gcis.get("gci3", []):
        c = iri_to_go(gci_class_name(o.filler)); d = iri_to_go(gci_class_name(o.superclass))
        if c and d and same_ns(c, d):
            ns = short_of[c]; per_ns[ns]["nf4"].append([c, d]); add_atoms(ns, c, d)

    for ns in ["cc", "mf", "bp"]:
        d = per_ns[ns]
        d["atoms"] = sorted(d["atoms"])
        path = os.path.join(args.out, f"{ns}.json")
        with open(path, "w") as fh:
            json.dump(d, fh)
        print(f"{ns}: atoms={len(d['atoms'])} nf1={len(d['nf1'])} nf2={len(d['nf2'])} "
              f"nf3={len(d['nf3'])} nf4={len(d['nf4'])} -> {path}", flush=True)


if __name__ == "__main__":
    main()
