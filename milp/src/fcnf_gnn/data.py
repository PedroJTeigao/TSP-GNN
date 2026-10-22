"""Save / load a set of labelled FCNF instances to one .npz.

Instances have different numbers of arcs, so arc arrays are concatenated and `arc_ptr` (num + 1,)
says where each instance starts: arcs of instance i are arc_ptr[i]:arc_ptr[i + 1].

Node arrays (all instances of a set have the same n): pos (num, n, 2), b (num, n).
Arc arrays (flat): src, dst (local node indices), c, F, u, length, x (optimal flow), y (open 0/1).
Per instance: ratio, cost, status (0 optimal, 1 time limit with a solution, 2 no solution),
mip_gap, bound, nodes, seconds, cap_rescaled.
"""

import numpy as np

STATUS_CODE = {"optimal": 0, "time_limit": 1}
NODE_KEYS = ("pos", "b")
ARC_KEYS = ("src", "dst", "c", "F", "u", "length", "x", "y")
INST_KEYS = ("ratio", "cost", "status", "mip_gap", "bound", "nodes", "seconds", "cap_rescaled")


def save_dataset(path, instances, **meta):
    """instances: list of dicts with the node, arc and per-instance keys above."""
    arrays = {k: np.stack([inst[k] for inst in instances]) for k in NODE_KEYS}
    arrays.update({k: np.concatenate([inst[k] for inst in instances]) for k in ARC_KEYS})
    arrays.update({k: np.array([inst[k] for inst in instances], dtype=float) for k in INST_KEYS})
    arrays["arc_ptr"] = np.concatenate([[0], np.cumsum([len(inst["src"]) for inst in instances])])
    np.savez_compressed(path, **arrays, **meta)


def load_dataset(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def num_instances(d):
    return len(d["arc_ptr"]) - 1


def get_instance(d, i):
    """Instance i of a loaded dataset as a dict (same keys as in save_dataset)."""
    a, z = int(d["arc_ptr"][i]), int(d["arc_ptr"][i + 1])
    inst = {k: d[k][i] for k in NODE_KEYS}
    inst.update({k: d[k][a:z] for k in ARC_KEYS if k in d})
    inst.update({k: d[k][i] for k in INST_KEYS if k in d})
    return inst
