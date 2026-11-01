"""Where robots lose time (experiment E3): read from the trajectory after the run, so the simulator stays out of it.

At each step a robot with a target (not in service, not at rest) does one of three things:
    progress   it gets closer to the target on the empty grid;
    wait       it stays in place;
    detour     it moves, but no closer.
Waits and detours are counted by the cell the robot stood in. The robot that blocked it is the one in a cell of the shortest
way, at that step or the next; that is one interaction of the pair.
"""

import numpy as np

KINDS = {"wait": 0, "detour": 1}


def analyse(wh, traj, targets):
    """{"waits", "detours", "visits": (H, W) counts; "events": (N, 5) t, cell, robot, kind, blocker (-1 if none);
    "pairs": (M, 3) i, j, interactions; "moving", "lost": robot-steps with a target, and those that were not progress}."""
    H, W = wh.height, wh.width
    waits, detours, visits = np.zeros((H, W), np.int32), np.zeros((H, W), np.int32), np.zeros((H, W), np.int32)
    events, pairs = [], {}
    moving = 0
    for t in range(len(targets)):
        now, nxt, goal = traj[t].tolist(), traj[t + 1].tolist(), targets[t].tolist()
        here = {c: i for i, c in enumerate(now)}
        there = {c: i for i, c in enumerate(nxt)}
        for i, c in enumerate(goal):
            if c < 0 or now[i] == c:
                continue
            moving += 1
            d = wh.dist_to(c)
            p, q = now[i], nxt[i]
            r = divmod(p, W)
            visits[r] += 1
            if d[q] < d[p]:
                continue
            kind = "wait" if q == p else "detour"
            (waits if q == p else detours)[r] += 1
            blocker = -1
            for u in wh.neighbours[p]:
                if d[u] < d[p]:
                    j = here.get(u, there.get(u, i))
                    if j != i:
                        blocker = j
                        break
            events.append((t, p, i, KINDS[kind], blocker))
            if blocker >= 0:
                key = (min(i, blocker), max(i, blocker))
                pairs[key] = pairs.get(key, 0) + 1
    return {"waits": waits, "detours": detours, "visits": visits, "moving": moving,
            "events": np.array(events, dtype=np.int32).reshape(-1, 5),
            "pairs": np.array([(a, b, n) for (a, b), n in sorted(pairs.items())], dtype=np.int32).reshape(-1, 3),
            "lost": int(waits.sum() + detours.sum())}


def hot_share(counts, free_mask, share=0.1):
    """Fraction of the counts in the hottest share of the free cells."""
    v = np.sort(counts[free_mask].ravel())[::-1]
    return float(v[:max(1, int(round(share * len(v))))].sum() / max(v.sum(), 1))


def hot_cells(counts, free_mask, share=0.1):
    """Boolean (H, W) map of the hottest share of the free cells."""
    idx = np.flatnonzero(free_mask.ravel())
    top = idx[np.argsort(-counts.ravel()[idx], kind="stable")[:max(1, int(round(share * len(idx))))]]
    out = np.zeros(free_mask.size, bool)
    out[top] = True
    return out.reshape(free_mask.shape)


def cell_time_share(events, cells, T, window=50, share=0.01):
    """(fraction of the waits in the hottest share of the (cell, window of time) boxes; fraction of the boxes with a wait)."""
    w = events[events[:, 3] == KINDS["wait"]]
    if not len(w):
        return 0.0, 0.0
    box = w[:, 1].astype(np.int64) * (T // window + 1) + w[:, 0] // window
    _, n = np.unique(box, return_counts=True)
    boxes = cells * (T // window + 1)
    top = np.sort(n)[::-1][:max(1, int(round(share * boxes)))]
    return float(top.sum() / n.sum()), len(n) / boxes
