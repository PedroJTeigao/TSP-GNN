"""The parsers on the development requests: per level, the share of exact rules, and the misses. No test text is read here.

    python scripts/phaseC_dev.py --level L2 --show 5
"""

import argparse
import json

import phaseC_language as Q
from warehouse import dsl as L
from warehouse import reqparser as R
from warehouse import world as Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="L1")
    p.add_argument("--arm", default="p1")
    p.add_argument("--split", default="dev")
    p.add_argument("--show", type=int, default=0)
    args = p.parse_args()
    assert args.level == "L1" or args.split == "dev", "the test paragraphs are not for the parser"
    para = Q.paragraphs()
    right = total = exact = rows = 0
    shown = 0
    for r in Q.requests(args.split):
        if args.level == "L2" and not para.get(r["idx"], {}).get("paragraph"):
            continue
        w = Y.make(r["seed"])
        ctx = L.Context(w)
        label = Q.labels(r, ctx)
        text = Q.text_of(r, args.level, para)
        obj = (R.parse_p0 if args.arm == "p0" else R.parse_p1)(text, w)
        got, errors = L.validate(obj, ctx)
        s = L.score(label, [q for x in got for q in x["prims"]])
        right += sum(s["correct"])
        total += len(s["correct"])
        exact += s["exact"]
        rows += 1
        if not s["exact"] and shown < args.show:
            shown += 1
            print(f"--- {r['idx']}  send={R.send_to_llm(text, w)}\n{text}")
            for x, ok in zip(r["rules"], s["correct"]):
                print(("  ok " if ok else "  NO ") + json.dumps(x["dsl"]))
            print("  got", json.dumps(obj["rules"]), errors)
    print(f"{args.arm} {args.level} {args.split}: rules {right}/{total}, requests {exact}/{rows}")


if __name__ == "__main__":
    main()
