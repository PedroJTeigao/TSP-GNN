"""Loading of the levels of text_v3_open (phase 5, step 3): each instance as text, legend, context and label."""

import json
import os

import numpy as np

from tsp_gnn import open_dsl as D
from tsp_gnn import open_gen as G
from tsp_gnn import rules as R

OUT = "results/phase5/open_language"
LEVELS = ("L0", "L1", "L2")


def _l0(n, num):
    rows = [json.loads(line) for line in open(f"data/phase5/text_rules_v2b_tsp{n}.jsonl")]
    arr = np.load(f"data/phase5/text_rules_tsp{n}.npz")
    bad = set(json.load(open("data/phase5/text_v2b_label_errors.json")).get(str(n), []))
    # same selection as step 2 (no label error, complete); the last 20 instances stay out, as in L1 / L2
    rows = [r for r in rows if r["v2_complete"] and r["idx"] not in bad and r["idx"] < 180]
    out = []
    for r in rows[:num]:
        coords = arr["coords"][r["idx"]]
        ctx = G.l0_context(r, coords, n)
        out.append({"idx": r["idx"], "level": "L0", "text": r["text_v2"], "legend": ctx.legend, "ctx": ctx,
                    "legend_text": D.NO_LEGEND, "label": G.l0_label(r, ctx), "coords": coords, "background": [],
                    "reference": arr["tours"][r["idx"]], "pure": arr["pure_tours"][r["idx"]]})
    return out


def _open(n, level, num, split):
    rows = [json.loads(line) for line in open(f"{OUT}/text_v3_open_tsp{n}.jsonl")]
    rows = [r for r in rows if r["split"] == split and r["feasible"]]
    arr = np.load(f"data/phase5/text_rules_tsp{n}.npz")
    tours = np.load(f"{OUT}/text_v3_open_tsp{n}.npz")
    tour = dict(zip(tours["idx"].tolist(), tours["tours"]))
    para = {}
    if level == "L2":
        para = {e["idx"]: e for e in map(json.loads, open(f"{OUT}/l2_tsp{n}.jsonl"))}
    out = []
    for r in rows:
        if level == "L2" and not para.get(r["idx"], {}).get("paragraph"):
            continue                                  # the model gave no text: nothing to translate
        coords = arr["coords"][r["idx"]]
        ctx = D.Context(r["legend"], coords, n)
        text = r["text_l1"] if level == "L1" else R.PREAMBLE + "\n" + para[r["idx"]]["paragraph"]
        out.append({"idx": r["idx"], "level": level, "text": text, "legend": r["legend"], "ctx": ctx,
                    "legend_text": D.legend_block(r["legend"]), "label": r["rules"], "coords": coords,
                    "background": r["background"], "reference": tour[r["idx"]], "pure": arr["pure_tours"][r["idx"]]})
    return out[:num] if num else out


def load_level(n, level, num=None, split="test"):
    return _l0(n, num) if level == "L0" else _open(n, level, num, split)


def label_prims(inst):
    return [p for r in inst["label"] for p in r["prims"]]


def audit_excluded():
    """(n, idx) of the L2 texts the manual audit found wrong, once the audit exists."""
    path = f"{OUT}/l2_audit.json"
    if not os.path.exists(path):
        return set()
    return {(e["n"], e["idx"]) for e in json.load(open(path))["items"] if not e["ok"]}
