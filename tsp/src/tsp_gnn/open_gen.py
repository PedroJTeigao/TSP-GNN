"""The text_v3_open data set (phase 5, step 3): legends and composite rules with exact labels.

Levels, all with labels that come from the formal generator (never from a model):
    L0  text_v2b, unchanged (control). Each primitive rule is one label rule.
    L1  composite rules from sentence templates: groups, quantifiers, exceptions, "the k closest", negation.
    L2  the same rules as L1, rewritten by gemma3:12b as running text (make_open_language.py). The rewrite is NOT
        filtered by any parser: its label errors are measured by a manual audit.

Legend: a table of named places (LEGEND_SIZE[n] customers), each with a name, a type, an area (a cell of a 3 x 3 grid
of the unit square) and coordinates. Group selectors refer to legend places only. It holds only part of the customers
because a full table at n = 1000 would not fit the 8192-token context of the LLMs. Names are street addresses, so the
type of a place can only be known from the legend.
"""

import numpy as np

from tsp_gnn import formalize as F
from tsp_gnn import open_dsl as D
from tsp_gnn import rules as R

SEED = 20261031
LEGEND_SIZE = {200: 40, 1000: 100}
AREAS = ["Alfama", "Belem", "Campolide", "Graca", "Lumiar", "Mouraria", "Olivais", "Restelo", "Telheiras"]
STREETS = ["Rua Azul", "Rua Verde", "Rua Nova", "Rua Larga", "Rua Alta", "Rua do Sol", "Rua da Paz", "Rua do Mar",
           "Avenida Central", "Avenida Norte", "Travessa Velha", "Largo do Carmo", "Calcada Branca", "Beco Escuro"]
KIND_WEIGHTS = {"hospital": 0.1, "school": 0.17, "warehouse": 0.13, "shop": 0.2, "house": 0.17, "pharmacy": 0.1,
                "office": 0.13}
PLURAL = {"hospital": "hospitals", "school": "schools", "warehouse": "warehouses", "shop": "shops", "house": "houses",
          "pharmacy": "pharmacies", "office": "offices"}
ARTICLE = {"hospital": "a", "school": "a", "warehouse": "a", "shop": "a", "house": "a", "pharmacy": "a",
           "office": "an"}
MAX_PAIRS = 150               # largest group product of a rule
DISTRACTORS = [
    "{name} has a new manager.",
    "{name} ordered extra packaging this week.",
    "The {kind_pl} in {area} were very busy last month.",
    "Most {kind_pl} pay at the end of the month.",
    "{name} was repainted recently, so it is easy to spot.",
    "The driver of today's shift is new to {area}.",
    "{name} and {name2} belong to the same owner.",
    "Last week the van reached {name} before {name2}, which was only a coincidence.",
    "Nobody complained about the road to {name} this year.",
    "The radio in the van is broken.",
    "Fuel prices went up this month.",
    "{name} asked for a copy of the receipt.",
]


def make_legend(n, idx, coords):
    """The legend of instance idx: places with id, name, kind, area and rounded coordinates."""
    rng = np.random.default_rng([SEED, n, idx, 1])
    ids = sorted(int(c) for c in rng.choice(np.arange(1, n), LEGEND_SIZE[n], replace=False))
    kinds = rng.choice(list(KIND_WEIGHTS), len(ids), p=list(KIND_WEIGHTS.values()))
    addresses = [f"{s} {k}" for s in STREETS for k in range(1, 30)]
    names = rng.choice(addresses, len(ids), replace=False)
    legend = []
    for c, kind, name in zip(ids, kinds, names):
        x, y = float(coords[c][0]), float(coords[c][1])
        area = AREAS[min(int(x * 3), 2) + 3 * min(int(y * 3), 2)]
        legend.append({"id": c, "name": str(name), "kind": str(kind), "area": area, "x": x, "y": y})
    return legend


def l0_context(row, coords, n):
    """Context of an L0 instance: its names dictionary, without types or areas."""
    legend = [{"id": int(c), "name": nm, "kind": "", "area": "", "x": 0.0, "y": 0.0} for c, nm in row["names"].items()]
    return D.Context(legend, coords, n)


def l0_label(row, ctx):
    """L0 label: one label rule per primitive rule of text_v2b."""
    out = []
    for r in row["rules"]:
        f, kind = r["formal"], r["type"]
        dsl = {"precedence": lambda: {"type": kind, "before": f["before"], "after": f["after"]},
               "forbidden_edge": lambda: {"type": kind, "a": f["i"], "b": f["j"]},
               "mandatory_edge": lambda: {"type": kind, "a": f["i"], "b": f["j"]},
               "time_window": lambda: {"type": kind, "who": f["customer"], "t1": f["t1"], "t2": f["t2"]},
               "soft_zone": lambda: {"type": kind, "zone": f["zone"]}}[kind]()
        out.append({"dsl": dsl, "prims": D.expand_rule(dsl, ctx), "shape": "L0"})
    return out


# ---------------------------------------------------------------------------------------------------- selector pool

def _named(place):
    return place["name"]


def _groups(legend):
    """(kind, members), (kind + area) groups of the legend with at least 2 members."""
    by_kind, by_both = {}, {}
    for p in legend:
        by_kind.setdefault(p["kind"], []).append(p)
        by_both.setdefault((p["kind"], p["area"]), []).append(p)
    return {k: v for k, v in by_kind.items() if len(v) >= 2}, {k: v for k, v in by_both.items() if len(v) >= 2}


def sel_kind(kind, ex=None):
    s = {"type_is": kind}
    if ex:
        s["except"] = [ex]
    return s


def _text_kind(kind, ex=None):
    return f"every {kind}" + (f" except {ex}" if ex else "")


def _sample_kind_pair(rng, kinds, used):
    names = [k for k in sorted(kinds) if k not in used]
    if len(names) < 2:
        return None
    a, b = rng.choice(names, 2, replace=False)
    return str(a), str(b)


def _exception(rng, members):
    return _named(members[int(rng.integers(len(members)))]) if len(members) >= 3 else None


def _nearest(rng, legend, k, kind=None):
    centre = legend[int(rng.integers(len(legend)))]
    s = {"nearest": {"k": k, "of": centre["name"]}}
    noun = f"{k} listed places"
    if kind:
        s["type_is"] = kind
        noun = f"{k} {PLURAL[kind]}" if k > 1 else f"{kind}"
    return s, centre, noun


# ---------------------------------------------------------------------------------------------------- rule shapes
# Each shape returns (dsl rule, sentence) or None. The sentence must say exactly what the rule says.

def prec_group(rng, ctx, kinds, areas, pure_len):
    pair = _sample_kind_pair(rng, kinds, ())
    if not pair:
        return None
    a, b = pair
    form = int(rng.integers(3))
    dsl = {"type": "precedence", "before": sel_kind(a), "after": sel_kind(b)}
    return dsl, [f"Visit every {a} before every {b}.",
                 f"No {b} may be served until every {a} has been visited.",
                 f"Every {b} can only be served after all the {PLURAL[a]} have been visited."][form]


def prec_except(rng, ctx, kinds, areas, pure_len):
    pair = _sample_kind_pair(rng, kinds, ())
    if not pair:
        return None
    a, b = pair
    side = int(rng.integers(2))
    ex = _exception(rng, kinds[a if side == 0 else b])
    if ex is None:
        return None
    if side == 0:
        dsl = {"type": "precedence", "before": sel_kind(a, ex), "after": sel_kind(b)}
        return dsl, [f"Every {a} except {ex} must be visited before every {b}.",
                     f"Apart from {ex}, all the {PLURAL[a]} come before all the {PLURAL[b]}."][int(rng.integers(2))]
    dsl = {"type": "precedence", "before": sel_kind(a), "after": sel_kind(b, ex)}
    return dsl, [f"Every {a} must be visited before every {b}, except {ex}.",
                 f"All the {PLURAL[b]} other than {ex} come after all the {PLURAL[a]}."][int(rng.integers(2))]


def prec_nearest(rng, ctx, kinds, areas, pure_len):
    k = int(rng.integers(2, 5))
    sel, centre, noun = _nearest(rng, ctx.legend, k)
    other = ctx.legend[int(rng.integers(len(ctx.legend)))]
    if other["id"] == centre["id"]:
        return None
    if int(rng.integers(2)):
        return ({"type": "precedence", "before": other["name"], "after": sel},
                f"{other['name']} must come before the {noun} closest to {centre['name']}.")
    return ({"type": "precedence", "before": sel, "after": other["name"]},
            f"Do not visit {other['name']} until the {noun} closest to {centre['name']} have been served.")


def prec_area(rng, ctx, kinds, areas, pure_len):
    if not areas:
        return None
    keys = sorted(areas)
    kind, area = keys[int(rng.integers(len(keys)))]
    other = ctx.legend[int(rng.integers(len(ctx.legend)))]
    if other["kind"] == kind and other["area"] == area:
        return None
    return ({"type": "precedence", "before": {"type_is": kind, "area_is": area}, "after": other["name"]},
            f"{other['name']} comes after every {kind} in {area}.")


def forb_group(rng, ctx, kinds, areas, pure_len):
    pair = _sample_kind_pair(rng, kinds, ())
    if not pair:
        return None
    a, b = pair
    dsl = {"type": "forbidden_edge", "a": sel_kind(a), "b": sel_kind(b)}
    return dsl, [f"Never drive directly between {ARTICLE[a]} {a} and {ARTICLE[b]} {b}.",
                 f"No {a} may be next to {ARTICLE[b]} {b} on the route.",
                 f"The direct legs between {PLURAL[a]} and {PLURAL[b]} are all closed."][int(rng.integers(3))]


def forb_area(rng, ctx, kinds, areas, pure_len):
    place = ctx.legend[int(rng.integers(len(ctx.legend)))]
    area = AREAS[int(rng.integers(len(AREAS)))]
    members = [p for p in ctx.legend if p["area"] == area and p["id"] != place["id"]]
    if not 2 <= len(members):
        return None
    return ({"type": "forbidden_edge", "a": place["name"], "b": {"area_is": area}},
            f"Do not go straight from {place['name']} to any place in {area}, or from any of them to it.")


def forb_except(rng, ctx, kinds, areas, pure_len):
    place = ctx.legend[int(rng.integers(len(ctx.legend)))]
    keys = [k for k in sorted(kinds) if k != place["kind"]]
    kind = keys[int(rng.integers(len(keys)))]
    ex = _exception(rng, kinds[kind])
    if ex is None:
        return None
    return ({"type": "forbidden_edge", "a": place["name"], "b": sel_kind(kind, ex)},
            f"{place['name']} may not be directly linked to any {kind} except {ex}.")


def mand_names(rng, ctx, kinds, areas, pure_len):
    a, b = rng.choice(len(ctx.legend), 2, replace=False)
    a, b = ctx.legend[int(a)]["name"], ctx.legend[int(b)]["name"]
    return ({"type": "mandatory_edge", "a": a, "b": b},
            [f"{a} and {b} must be served one right after the other, in either order.",
             f"The route has to contain the direct leg between {a} and {b}."][int(rng.integers(2))])


def mand_nearest(rng, ctx, kinds, areas, pure_len):
    keys = sorted(kinds)
    kind = keys[int(rng.integers(len(keys)))] if int(rng.integers(2)) else None
    sel, centre, noun = _nearest(rng, ctx.legend, 1, kind)
    b = sel
    if kind:
        return ({"type": "mandatory_edge", "a": centre["name"], "b": b},
                f"{centre['name']} must be directly next to the closest {kind} on the route.")
    return ({"type": "mandatory_edge", "a": centre["name"], "b": b},
            f"Go straight between {centre['name']} and the listed place closest to it, in either direction.")


def _window(rng, pure_len):
    c = float(rng.uniform(0.3, 0.7)) * pure_len
    return round(c - 0.2 * pure_len, 2), round(c + 0.2 * pure_len, 2)


def tw_area(rng, ctx, kinds, areas, pure_len):
    keys = [k for k in sorted(areas) if len(areas[k]) <= 4]
    if not keys:
        return None
    kind, area = keys[int(rng.integers(len(keys)))]
    t1, t2 = _window(rng, pure_len)
    return ({"type": "time_window", "who": {"type_is": kind, "area_is": area}, "t1": t1, "t2": t2},
            [f"Every {kind} in {area} must be served between {t1:.2f} and {t2:.2f}.",
             f"Do not serve any {kind} in {area} before {t1:.2f} or after {t2:.2f}."][int(rng.integers(2))])


def tw_nearest(rng, ctx, kinds, areas, pure_len):
    k = int(rng.integers(2, 4))
    sel, centre, noun = _nearest(rng, ctx.legend, k)
    t1, t2 = _window(rng, pure_len)
    return ({"type": "time_window", "who": sel, "t1": t1, "t2": t2},
            f"The {noun} closest to {centre['name']} are open from {t1:.2f} to {t2:.2f} only.")


def tw_except(rng, ctx, kinds, areas, pure_len):
    keys = [k for k in sorted(areas) if 3 <= len(areas[k]) <= 5]
    if not keys:
        return None
    kind, area = keys[int(rng.integers(len(keys)))]
    ex = _exception(rng, areas[(kind, area)])
    t1, t2 = _window(rng, pure_len)
    return ({"type": "time_window", "who": {"type_is": kind, "area_is": area, "except": [ex]}, "t1": t1, "t2": t2},
            f"Every {kind} in {area} except {ex} must be served between {t1:.2f} and {t2:.2f}.")


def tw_union(rng, ctx, kinds, areas, pure_len):
    a, b = rng.choice(len(ctx.legend), 2, replace=False)
    a, b = ctx.legend[int(a)]["name"], ctx.legend[int(b)]["name"]
    t1, t2 = _window(rng, pure_len)
    return ({"type": "time_window", "who": {"any_of": [a, b]}, "t1": t1, "t2": t2},
            f"Both {a} and {b} can only be served from {t1:.2f} to {t2:.2f}.")


def soft(rng, ctx, kinds, areas, pure_len):
    zone = sorted(R.ZONES)[int(rng.integers(len(R.ZONES)))]
    return ({"type": "soft_zone", "zone": zone},
            [f"Avoid the {zone} zone if possible.", f"Keep the driving inside the {zone} zone to a minimum."]
            [int(rng.integers(2))])


SHAPES = {"precedence": [prec_group, prec_except, prec_nearest, prec_area],
          "forbidden_edge": [forb_group, forb_area, forb_except],
          "mandatory_edge": [mand_names, mand_nearest],
          "time_window": [tw_area, tw_nearest, tw_except, tw_union],
          "soft_zone": [soft]}


def sample_rules(rng, legend, coords, n, pure_len, k):
    """k composite rules that pass the code check together (validate_for_solver accepts all of them), never a window
    with a precedence, at most 2 mandatory edges with a precedence or window (LKH-3 runs once per direction)."""
    ctx = D.Context(legend, coords, n)
    kinds, areas = _groups(legend)
    types = R.sample_types(rng, k)
    out = []
    for kind in types:
        for attempt in range(400):
            if attempt == 200:                 # e.g. a third mandatory rule cannot fit: this slot gets a forbidden edge
                kind = "forbidden_edge"
            shapes = SHAPES[kind]
            shape = shapes[int(rng.integers(len(shapes)))]
            got = shape(rng, ctx, kinds, areas, pure_len)
            if got is None:
                continue
            dsl, sentence = got
            try:
                prims = D.expand_rule(dsl, ctx)
            except D.DslError:
                continue
            cand = out + [{"dsl": dsl, "prims": prims, "sentence": sentence, "shape": shape.__name__}]
            if len(prims) > MAX_PAIRS or D.keyset(prims) & set().union(*[D.keyset(r["prims"]) for r in out] or [set()]):
                continue
            if len({F.rule_key(p)[0] for p in prims}) != 1:
                continue
            _, rejected = D.validate_for_solver(cand, coords)
            mand = sum(p["type"] == "mandatory_edge" for r in cand for p in r["prims"])
            if rejected or mand > 2:
                continue
            out = cand
            break
        else:
            raise RuntimeError(f"could not sample a {kind} rule")
    return out


def distractors(rng, legend, k):
    """k background sentences with no rule, some of them naming places of the legend."""
    out = []
    for t in rng.choice(DISTRACTORS, k, replace=False):
        a, b = rng.choice(len(legend), 2, replace=False)
        kind = str(rng.choice(list(KIND_WEIGHTS)))
        out.append(str(t).format(name=legend[int(a)]["name"], name2=legend[int(b)]["name"], kind_pl=PLURAL[kind],
                                 area=legend[int(a)]["area"]))
    return out


L2_PROMPT = """You are a dispatcher writing today's briefing for a van driver.
Rewrite the rules below as ONE flowing paragraph of plain English, the way a person would write it.

Rules to include (every one of them, with exactly the same meaning):
{rules}

Background remarks to weave in among the rules (they are not rules; keep their meaning):
{distractors}

Requirements:
- Plain running text: no lists, no headings, no line breaks.
- Do not change, add or drop any place name, type, area, zone name, number or time, and do not add new conditions.
- Do not repeat a sentence as given: change the wording and the order of the words.
- When it reads naturally, refer back to something you have just mentioned ("the latter", "that warehouse", "the same
  place") instead of repeating its name, as long as it stays unambiguous.
- Where it fits, state one rule with a double negation ("it is not acceptable to skip ...", "no ... unless ...").
Write only the paragraph."""


def l2_prompt(sentences, background):
    return L2_PROMPT.format(rules="\n".join(f"- {s}" for s in sentences),
                            distractors="\n".join(f"- {s}" for s in background))
