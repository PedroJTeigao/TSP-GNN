"""FCNF instance -> torch_geometric Data.

    x          (n, 5)   node features: b_i / max|b|, is supply, is demand, in-degree, out-degree
    edge_index (2, E)   message-passing graph: every arc in both directions, without duplicates
                        (an arc and its reverse arc give one undirected edge)
    arc_index  (2, m)   the arcs themselves, row 0 = tail i, row 1 = head j (the head scores these)
    arc_attr   (m, 5)   arc features: c, F, u, length divided by their mean in the instance, and
                        log(ratio) (ratio = F / (c u), the same on every arc; not normalized)
    y          (m,)     1 if the arc is open in the reference solution
PyG increments every attribute whose name contains "index" by the number of nodes when it
batches graphs, so arc_index stays valid in a batch.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from fcnf_gnn.data import get_instance, load_dataset, num_instances

NODE_FEATURES = 5
ARC_FEATURES = 5


def node_features(inst):
    b = np.asarray(inst["b"], float)
    n = len(b)
    indeg = np.bincount(inst["dst"], minlength=n)
    outdeg = np.bincount(inst["src"], minlength=n)
    return np.stack([b / np.abs(b).max(), b > 0, b < 0, indeg, outdeg], axis=1).astype(np.float32)


def arc_features(inst):
    cols = [np.asarray(inst[k], float) / np.mean(inst[k]) for k in ("c", "F", "u", "length")]
    ratio = np.mean(np.asarray(inst["F"]) / (np.asarray(inst["c"]) * np.asarray(inst["u"])))
    cols.append(np.full(len(inst["src"]), np.log(ratio)))
    return np.stack(cols, axis=1).astype(np.float32)


def build_graph(inst):
    n = len(inst["b"])
    arc_index = torch.as_tensor(np.stack([inst["src"], inst["dst"]]), dtype=torch.long)
    data = Data(x=torch.from_numpy(node_features(inst)),
                edge_index=to_undirected(arc_index, num_nodes=n),
                arc_index=arc_index, arc_attr=torch.from_numpy(arc_features(inst)), num_nodes=n)
    if "y" in inst:
        data.y = torch.as_tensor(np.asarray(inst["y"]), dtype=torch.float32)
    return data


def load_graphs(path, limit=None):
    d = load_dataset(path)
    num = num_instances(d) if limit is None else min(limit, num_instances(d))
    return [build_graph(get_instance(d, i)) for i in range(num)]
