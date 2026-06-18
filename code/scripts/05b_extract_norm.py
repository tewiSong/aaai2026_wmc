"""Task A (no mowl): extract the EL++ normal forms from the pre-normalized GO dataset.

The dataset at /ibex/user/songt/datasets/GO/ is the EL++ normalization of GO (produced by a
mowl/jcel pipeline), stored as an ML benchmark: each sample is one normalized axiom with the
left-hand side(s) and the right-hand side given as the answer candidate (carrying its URI in
`extra.uri`). We reconstruct the clean axioms across all splits (train+val+test =
71331+10206+17727+10202 = 109466 axioms) and partition them by namespace:

  NF1  lhs sub rhs                 -> nf1 (lhs, rhs)
  NF2  conj0 & conj1 sub rhs       -> nf2 (conj0, conj1, rhs)
  NF3  lhs sub exists role.rhs     -> nf3 (lhs, rhs)
  NF4  exists role.lhs sub rhs     -> nf4 (lhs, rhs)

Writes <out>/<ns>.json = {atoms, nf1, nf2, nf3, nf4} for ns in cc/mf/bp.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

GO_DIR = "/ibex/user/songt/datasets/GO"
OBO_PREFIX = "http://purl.obolibrary.org/obo/"
NS_SHORT = {"cellular_component": "cc", "molecular_function": "mf", "biological_process": "bp"}


def iri_to_go(iri):
    if iri and iri.startswith(OBO_PREFIX):
        local = iri[len(OBO_PREFIX):]
        if local.startswith("GO_"):
            return "GO:" + local[3:]
    return None


def parse_namespaces(obo_path):
    ns, cur, in_term = {}, None, False
    with open(obo_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                in_term, cur = True, None
            elif line.startswith("["):
                in_term = False
            elif in_term and line.startswith("id:"):
                cur = line.split(":", 1)[1].strip()
            elif in_term and line.startswith("namespace:") and cur:
                ns[cur] = line.split(":", 1)[1].strip()
    return ns


def answer_uri(sample):
    cand = sample["candidates"][sample["answer_option"] - 1]
    return cand.get("extra", {}).get("uri")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", default="data/go-basic.obo")
    ap.add_argument("--out", default="data/processed/norm")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ns_of = parse_namespaces(args.obo)
    short_of = {gid: NS_SHORT[v] for gid, v in ns_of.items() if v in NS_SHORT}

    per = defaultdict(lambda: {"atoms": set(), "nf1": [], "nf2": [], "nf3": [], "nf4": []})
    counts = defaultdict(int)
    dropped = 0

    def same_ns(*gids):
        s = {short_of.get(g) for g in gids}
        return len(s) == 1 and None not in s

    for split in ["train", "val", "test"]:
        data = json.load(open(os.path.join(GO_DIR, f"{split}_samples.json")))
        for s in data:
            t = s["nf_type"]
            rhs = iri_to_go(answer_uri(s))
            if t == "NF1":
                lhs = iri_to_go(s["lhs"]["uri"])
                if lhs and rhs and same_ns(lhs, rhs):
                    ns = short_of[lhs]; per[ns]["nf1"].append([lhs, rhs])
                    per[ns]["atoms"].update([lhs, rhs]); counts["nf1"] += 1
                else:
                    dropped += 1
            elif t == "NF2":
                a = iri_to_go(s["conjuncts"][0]["uri"]); b = iri_to_go(s["conjuncts"][1]["uri"])
                if a and b and rhs and same_ns(a, b, rhs):
                    ns = short_of[a]; per[ns]["nf2"].append([a, b, rhs])
                    per[ns]["atoms"].update([a, b, rhs]); counts["nf2"] += 1
                else:
                    dropped += 1
            elif t == "NF3":
                lhs = iri_to_go(s["lhs"]["uri"])
                if lhs and rhs and same_ns(lhs, rhs):
                    ns = short_of[lhs]; per[ns]["nf3"].append([lhs, rhs])
                    per[ns]["atoms"].update([lhs, rhs]); counts["nf3"] += 1
                else:
                    dropped += 1
            elif t == "NF4":
                lhs = iri_to_go(s["lhs"]["uri"])
                if lhs and rhs and same_ns(lhs, rhs):
                    ns = short_of[lhs]; per[ns]["nf4"].append([lhs, rhs])
                    per[ns]["atoms"].update([lhs, rhs]); counts["nf4"] += 1
                else:
                    dropped += 1

    print("kept axioms:", dict(counts), "dropped (cross-ns / non-GO):", dropped)
    for ns in ["cc", "mf", "bp"]:
        d = per[ns]; d["atoms"] = sorted(d["atoms"])
        with open(os.path.join(args.out, f"{ns}.json"), "w") as fh:
            json.dump(d, fh)
        print(f"{ns}: atoms={len(d['atoms'])} nf1={len(d['nf1'])} nf2={len(d['nf2'])} "
              f"nf3={len(d['nf3'])} nf4={len(d['nf4'])}")


if __name__ == "__main__":
    main()
