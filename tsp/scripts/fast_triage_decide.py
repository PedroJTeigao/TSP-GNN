"""Fast triage: does version B go on to the policy runs? Only a B score whose mean Spearman correlation with the
current per-window score is >= 0.5 on every checked set (scripts/fast_triage_check.py). Writes
results/phase3_fast_triage/b_decision.json; exit code 0 always (the runner reads the file)."""

import json
import os

ROOT = "results/phase3_fast_triage"
SETS = ("tsp1000_test", "tsp10000_test_concorde")
THRESHOLD = 0.5

out = {}
for v in ("B disagreement", "B learned"):
    rho = {}
    for name in SETS:
        s = json.load(open(os.path.join(ROOT, "check", f"{name}.json")))["summary"]
        if v in s:
            rho[name] = s[v]["spearman"]["mean"]
    out[v] = {"spearman_mean": rho, "threshold": THRESHOLD,
              "pass": len(rho) == len(SETS) and min(rho.values()) >= THRESHOLD}
with open(os.path.join(ROOT, "b_decision.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
for v, d in out.items():
    print(f"{v.split()[1]}={'yes' if d['pass'] else 'no'}")
