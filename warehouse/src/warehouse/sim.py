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

from warehouse import dispatch as D
from warehouse import env as W
from warehouse import mapf as M
from warehouse import sequencing as S

SERVICE = 3                                     # steps at a shelf and at a station


def run_episode(wh, cat, orders, method, planner, seed, service=SERVICE, max_steps=30000, stop_at=None, robots=None,
                planner_args=None, dispatcher=None, metric=None, cache=None):
    """Runs until every order is delivered, or until stop_at (lifelong mode), or max_steps. Returns the episode: traj (T+1,
    R) cells; targets (T, R) the cell each robot is heading to, -1 at rest, -2 in service; events; step_seconds; stats.
    dispatcher: who takes which orders (default: dispatch.Nearest). metric: the distances of layer 1, with dist_to like the
    Warehouse (default: the empty grid); planners always use the empty grid. cache: visit orders kept across episodes."""
    R = robots or wh.layout.robots
    metric = metric or wh
    dispatcher = dispatcher or D.Nearest(wh, metric)
    pos = W.start_cells(wh, R, seed)
    home = list(pos)
    plan = M.make_planner(planner, wh, seed=seed, **(planner_args or {}))
    pending = deque()
    future = deque(sorted(orders, key=lambda o: (o.arrival, o.oid)))
    queue = [[] for _ in range(R)]                  # next stops: (cell, dwell, kind, order id, shelf)
    hold, age, busy = [0] * R, [0] * R, [None] * R  # busy[i]: orders robot i still has to deliver
    started, carried = [False] * R, [()] * R        # it has begun to pick; the orders it took
    traj, targets, secs, events = [list(pos)], [], [], []
    current = [None] * R                            # the event in progress
    buffered = deque()
    cache = {} if cache is None else cache
    seq_rows, seq_seconds = [], 0.0
    delivered = 0
    changed = True                                  # a robot came free or an order arrived, since the last assignment

    def arrive(i, t):
        """Starts the service of robot i if it stands on its next stop; the stops of a robot can share a cell."""
        while queue[i] and hold[i] == 0 and queue[i][0][0] == pos[i]:
            cell, dwell, kind, oid, shelf = queue[i].pop(0)
            age[i] = 0
            if dwell == 0:
                continue
            hold[i] = dwell
            started[i] = started[i] or kind == "pick"
            current[i] = {"kind": kind, "robot": i, "order": oid, "shelf": shelf, "cell": cell, "start": t, "end": t + dwell}

    observe = getattr(dispatcher, "observe", lambda *_: None)

    def give(r, group, t):
        nonlocal seq_seconds
        stops, deliveries, sec = S.route(metric, wh, pos[r], group, method, cache)
        seq_seconds += sec
        busy[r], carried[r], started[r] = len(group), group, False
        queue[r] = [(wh.access[s], service, "pick", owner, s) for s, owner in stops]
        queue[r] += [(cell, service, "deliver", oid, None) for cell, oid in deliveries] + [(home[r], 0, "rest", None, None)]
        events.extend({"kind": "assign", "robot": r, "order": o.oid, "t": t} for o in group)
        seq_rows.append({"order": group[0].oid, "stops": len(stops), "seconds": sec})
        age[r] = 0
        arrive(r, t)

    def assign(t):
        nonlocal changed
        while future and future[0].arrival <= t:
            pending.append(future.popleft())
            changed = True
        free = [i for i in range(R) if busy[i] is None and hold[i] == 0]
        back = {}
        if dispatcher.reassign and changed:         # a robot that has not started to pick can still change its order
            for i in range(R):
                if busy[i] is not None and not started[i] and hold[i] == 0:
                    back[i] = carried[i]
                    pending.extendleft(carried[i])
                    busy[i] = None
                    free.append(i)
        changed = False
        for r, group in dispatcher.choose(t, free, pending, pos):
            if back.pop(r, None) != group:
                give(r, group, t)
            else:
                busy[r] = len(group)                # the same order: its route stays
        for r in back:                              # no order for it now
            queue[r], carried[r] = [(home[r], 0, "rest", None, None)], ()

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
        observe(t, pos, nxt, targets[-1])
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
                    owners = ev["order"] if isinstance(ev["order"], tuple) else (ev["order"],)
                    events.extend({**ev, "order": o} for o in owners)    # a shelf shared by two orders serves both
                    if ev["kind"] == "deliver":
                        busy[i] -= 1
                        busy[i] = busy[i] or None
                        delivered += 1
                        changed = True
            elif pos[i] != home[i] and not queue[i]:     # pushed off its rest cell
                queue[i] = [(home[i], 0, "rest", None, None)]
            arrive(i, t)
    return {"traj": np.array(traj, dtype=np.int32), "targets": np.array(targets, dtype=np.int32).reshape(-1, R),
            "events": events, "step_seconds": np.array(secs), "seq": seq_rows, "seq_seconds": seq_seconds,
            "delivered": delivered, "pending_end": len(pending) + len(future),
            "planner": {k: getattr(plan, k, 0) for k in ("calls", "fallbacks", "timeouts", "expanded", "spent")},
            "service": service, "seconds": float(np.sum(secs))}
