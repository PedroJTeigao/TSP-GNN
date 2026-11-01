"""The simulator: orders -> robots -> visit order -> planner -> trajectory.

Each step: arrived orders go to free robots (layer 1), the planner gives the next cells if there is no plan left (layer 2),
the robots move, and services advance. A robot that reaches its next stop stays dwell steps. Only what happened is written:
the trajectory, the events and the time per step. The verifier reads those and nothing else.

Time per step is the computation of that step (assignment, visit order, planning); a planner with a window spends it in
its planning steps, so its median can be 0 and the mean or the p95 is the fair figure.
"""

import time
from collections import deque

import numpy as np

from warehouse import env as W
from warehouse import mapf as M
from warehouse import sequencing as S

SERVICE = 3                                     # steps at a shelf and at a station


def run_episode(wh, cat, orders, method, planner, seed, service=SERVICE, max_steps=30000, stop_at=None, robots=None,
                planner_args=None):
    """Runs until every order is delivered, or until stop_at (lifelong mode), or max_steps. Returns the episode: traj (T+1,
    R) cells; targets (T, R) the cell each robot is heading to, -1 at rest, -2 in service; events; step_seconds; stats."""
    R = robots or wh.layout.robots
    pos = W.start_cells(wh, R, seed)
    home = list(pos)
    plan = M.make_planner(planner, wh, seed=seed, **(planner_args or {}))
    pending = deque()
    future = deque(sorted(orders, key=lambda o: (o.arrival, o.oid)))
    queue = [[] for _ in range(R)]                  # next stops: (cell, dwell, kind, order id, shelf)
    hold, age, busy = [0] * R, [0] * R, [None] * R
    traj, targets, secs, events = [list(pos)], [], [], []
    current = [None] * R                            # the event in progress
    buffered = deque()
    cache, seq_rows, seq_seconds = {}, [], 0.0
    delivered = 0

    def arrive(i, t):
        """Starts the service of robot i if it stands on its next stop; the stops of a robot can share a cell."""
        while queue[i] and hold[i] == 0 and queue[i][0][0] == pos[i]:
            cell, dwell, kind, oid, shelf = queue[i].pop(0)
            age[i] = 0
            if dwell == 0:
                continue
            hold[i] = dwell
            current[i] = {"kind": kind, "robot": i, "order": oid, "shelf": shelf, "cell": cell, "start": t, "end": t + dwell}

    def assign(t):
        nonlocal seq_seconds
        while future and future[0].arrival <= t:
            pending.append(future.popleft())
        free = [i for i in range(R) if busy[i] is None and hold[i] == 0]
        while pending and free:
            o = pending.popleft()
            stops = [wh.access[s] for s in o.shelves]
            r = S.assign_robot(wh, free, pos, stops)
            free.remove(r)
            idx, sec = S.sequence(wh, pos[r], stops, o.station, method, cache)
            seq_seconds += sec
            shelves = [o.shelves[k] for k in idx]
            busy[r] = o.oid
            queue[r] = [(wh.access[s], service, "pick", o.oid, s) for s in shelves]
            queue[r] += [(o.station, service, "deliver", o.oid, None), (home[r], 0, "rest", None, None)]
            events.append({"kind": "assign", "robot": r, "order": o.oid, "t": t})
            seq_rows.append({"order": o.oid, "stops": len(stops), "seconds": sec})
            age[r] = 0
            arrive(r, t)

    t = 0
    total = len(orders)
    while t < max_steps:
        if stop_at is None and delivered == total:
            break
        if stop_at is not None and t >= stop_at:
            break
        tick = time.perf_counter()
        if not buffered:                            # orders are given out only between plans, which they would change
            assign(t)
            state = [[(c, d) for c, d, *_ in q] for q in queue]
            paths = plan.plan(t, pos, hold, state, age)
            buffered.extend(zip(*paths))
        nxt = list(buffered.popleft())
        secs.append(time.perf_counter() - tick)
        targets.append([-2 if hold[i] > 0 else (queue[i][0][0] if queue[i] else -1) for i in range(R)])
        for i in range(R):
            assert hold[i] == 0 or nxt[i] == pos[i], f"robot {i} left its service at step {t}"
            if queue[i] and hold[i] == 0 and busy[i] is not None:
                age[i] += 1
        pos = nxt
        t += 1
        traj.append(list(pos))
        for i in range(R):
            if hold[i] > 0:
                hold[i] -= 1
                if hold[i] == 0:
                    ev = current[i]
                    current[i] = None
                    events.append(ev)
                    if ev["kind"] == "deliver":
                        busy[i] = None
                        delivered += 1
            elif pos[i] != home[i] and not queue[i]:     # pushed off its rest cell
                queue[i] = [(home[i], 0, "rest", None, None)]
            arrive(i, t)
    return {"traj": np.array(traj, dtype=np.int32), "targets": np.array(targets, dtype=np.int32).reshape(-1, R),
            "events": events, "step_seconds": np.array(secs), "seq": seq_rows, "seq_seconds": seq_seconds,
            "delivered": delivered, "pending_end": len(pending) + len(future),
            "planner": {k: getattr(plan, k, 0) for k in ("calls", "fallbacks", "timeouts", "expanded", "spent")},
            "service": service, "seconds": float(np.sum(secs))}
