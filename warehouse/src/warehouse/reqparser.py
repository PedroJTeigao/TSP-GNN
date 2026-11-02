"""Rule-based translation of warehouse requests into the DSL: no LLM.

    p0  the keyword parser of tsp_gnn.keyword_parser carried over: one rule per sentence, the type from a keyword, the selector from the
        first mentions. It has no exceptions, no coreference, no filter for background remarks. The old parser itself reads customers
        and times of the TSP, so only its method is kept.
    p1  the fair non-LLM baseline, in the steps of tsp_gnn.open_parser, written in about 2 h from the L1 templates and the held-out
        development paragraphs of L2, and frozen before the audit:
        1. the text is cut into sentences and each sentence into clauses at semicolons and at ", and" / ", but" / ", while" / ", so",
           when both sides say something;
        2. each clause is scanned for mentions: customers, order numbers, corridors, stations, kinds, urgent, robot types, steps;
        3. "except / apart from / other than / but not" attaches what follows to the group before it, or to the only group of the clause;
        4. the type of the rule comes from keywords, in this order: closure -> robot type -> trips kept apart -> deadline -> station ->
           pick order;
        5. the direction of a pick order comes from the order word and the position of the two sides;
        6. coreference: "that corridor", "the same customer", "they", "those", "their" point to the last mention of that sort;
        7. remarks about the past give no rule.
send_to_llm is the trigger of the hybrid h1: it reads only the text and the output of p1.
"""

import re

from warehouse import dsl as L
from warehouse import world as Y

KINDS = "|".join(Y.SPECIAL + ("normal",))
PAST = re.compile(r"\b(last (?:week|month|year|night|spring|winter|summer)|yesterday|recently|on sunday|in the spring|coincidence|was|were|had|has changed|called|paid|sold|share the same|"
                  r"accountant|updated|repaint\w*|cleaned|cleaning|finished|wrapped|thank(?:ed|s)?|invoice|fluke|fresh coat|looks? great|last week's)\b", re.I)
MUST = re.compile(r"\b(must|may not|never|cannot|has to|have to|needs? to|may only)\b", re.I)
CLOSE = re.compile(r"\b(clos(?:ed|es?|ure)|out of use|out of service|out of bounds|off[- ]limits|keep (?:the |all )?robots? (?:out|away)|stay (?:out|away)|nobody may|no one may|no robot|no robots|"
                   r"only opens?|reopens?|resurfac\w*|repair\w*|maintenance|blocked|impassable|no way through|inaccessible|unavailable|shut|not (?:be )?(?:use|used|usable|available|accessible|open)|"
                   r"cannot be used|can't be used|sealed|work(?:s|ing)? in|works? on|being (?:re\w+|fix\w+|clean\w+|paint\w+)|cordoned|barred|avoid|strictly|offline|shouldn't be operating|not be operating)\b", re.I)
APART = re.compile(r"\b(share (?:a|the same|one) trip|sharing (?:a|the same|one) trip|same trip|one trip|together|mix(?:ed|ing)?|go(?:es)? with|travel(?:s|ling|ing)? with|alongside|combin\w+|keep\w* \w+ apart|"
                   r"separate\w*|kept apart|on a single trip|apart(?! from))\b", re.I)
DELIVER = re.compile(r"\b(?:by|before|no later than|not later than|at the latest by|until|after|within|prior to|deadline(?: of| is)?|cut-?off(?: of| is)?|ahead of)\s+(?:step\s+)?(\d+)\b", re.I)
PICK = re.compile(r"\b(pick\w*|collect\w*|gather\w*|grab\w*|retriev\w*|last|first|start with|begin with|end with|finish with|for the end|ahead of|anything else|before|after|precede\w*|follow\w*)\b", re.I)
STATION_WORD = re.compile(r"\b(direct\w*|designated|conclude\w*|handed|complet\w*|deliver\w*|end(?:s|ing)?|go(?:es)?|takes?|sent?|routed?|head\w*|arriv\w*|destined|bound|handle\w*|drop\w*|receive\w*|finish\w*|must reach)\b", re.I)
EXCEPT = re.compile(r"\b(except(?: for)?|excepting|apart from|aside from|other than|excluding|but not|save for|barring|with the (?:sole |only |single )?exception of|not including|unless)\b", re.I)
SPLIT = re.compile(r"\s*;\s*|\s+[–—-]\s+|\s*[,]\s*(?:and|but|while|whereas|plus|so|as|also|though|although|however|meanwhile)\s+|\.\s+(?=[A-Z])|\s*:\s+")
WORDS = re.compile(r"\b(must|may|cannot|can't|never|only|no|not|has to|have to|needs?|should|closed|closure|keep|by step|no later|last|first|before|after|together|share|mix|type [AB]|station [A-F])\b", re.I)


SYNONYMS = [(r"\b(?:delicate|breakable)\b", "fragile"), (r"\bheavier\b", "heavy"), (r"\b(?:chilled|cold|refrigerated)\b", "frozen"), (r"\bother clients\b", "other customers")]
VAN = re.compile(r"\b(?:leav\w*|depart\w*|scheduled for|through)\b[^.;]{0,40}?\bstep\s+(\d+)", re.I)


def normalise(text):
    """Spelling of the writer of L2: other words for the kinds."""
    text = text.replace("\u2019", "'")
    for a, b in SYNONYMS:
        text = re.sub(a, b, text, flags=re.I)
    return text


def sentences(text):
    out = []
    for line in normalise(text).splitlines():
        line = line.strip()
        line = line[2:] if line.startswith("- ") else line
        out += [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", line) if p.strip()]
    return out


class Mentions:
    def __init__(self, world):
        self.w = world
        names = sorted(world.customers, key=len, reverse=True)
        self.name_re = re.compile("|".join(re.escape(x) for x in names), re.I)
        self.proper = {x.lower(): x for x in names}
        self.last = {}

    def scan(self, clause):
        """[(start, end, kind of mention, value)] in order."""
        out, taken = [], []

        def add(s, e, what, value):
            if not any(s < b and a < e for a, b in taken):
                taken.append((s, e))
                out.append((s, e, what, value))
        for m in self.name_re.finditer(clause):
            add(m.start(), m.end(), "customer", self.proper[m.group().lower()])
        for m in re.finditer(r"\bcorridors?\s+(\d+)(?:(?:\s*,\s*|\s+and\s+|\s*,\s*and\s+)(?:corridor\s+)?(\d+))*", clause, re.I):
            add(m.start(), m.end(), "corridor", [int(x) for x in re.findall(r"\d+", m.group())])
        for m in re.finditer(r"\borders?\s+(?:number\s+|no\.?\s+|#)?(\d+)\b", clause, re.I):
            add(m.start(), m.end(), "order", int(m.group(1)))
        for m in re.finditer(r"\b[Ss]tations?\s+([A-F])\b", clause):
            add(m.start(), m.end(), "station", m.group(1))
        for m in re.finditer(r"\b[Tt]ype[- ]([AB])\b|\b([AB])[- ]type\b", clause):
            add(m.start(), m.end(), "robot", m.group(1) or m.group(2))
        for m in re.finditer(rf"\b({KINDS})\b", clause, re.I):
            add(m.start(), m.end(), "kind", m.group(1).lower())
        for m in re.finditer(r"\burgent\b", clause, re.I):
            add(m.start(), m.end(), "priority", "urgent")
        out.sort(key=lambda x: x[0])
        return out

    def coref(self, clause, ms):
        """That corridor, the same customer, they, those: the last mention of that sort."""
        have = {m[2] for m in ms}
        extra = []
        for m in re.finditer(r"\b(?:that|the same|this|said) (corridor|customer|station|order|client)\b", clause, re.I):
            what = {"client": "customer"}.get(m.group(1).lower(), m.group(1).lower())
            if what in self.last and what not in have:
                extra.append((m.start(), m.end(), what, self.last[what]))
        if not any(w in have for w in ("customer", "kind", "order", "priority", "station")) and "who" in self.last:
            m = re.search(r"\b(they|them|those|these|their|the latter|the former|its|it|her|his|that one)\b", clause, re.I)
            if m:
                what, value = self.last["who"]
                extra.append((m.start(), m.end(), what, value))
        kinds = [m for m in ms if m[2] == "kind"]
        order_word = re.search(r"\b(before|ahead of|after)\b", clause, re.I)
        if order_word and len(kinds) == 1 and "kind" in self.last and self.last["kind"] != kinds[0][3]:     # fragile goods never come before them
            m = re.search(r"\b(them|those|these)\b", clause, re.I)
            if m:
                extra.append((m.start(), m.end(), "kind", self.last["kind"]))
        return sorted(ms + extra, key=lambda x: x[0])

    def remember(self, ms):
        for _, _, what, value in ms:
            self.last[what] = value
            if what in ("customer", "kind", "order", "priority"):
                self.last["who"] = (what, value)


def split_except(ms, clause):
    """(main mentions, excepted). An exception runs to the end of the clause, or to the comma if it opens the clause. A kind repeated in the
    exception ("except the heavy orders of X") only says which group it belongs to."""
    ex = EXCEPT.search(clause)
    if not ex:
        return ms, []
    end = len(clause)
    if not clause[:ex.start()].strip(" ,") or not [m for m in ms if m[0] < ex.start()]:
        comma = clause.find(",", ex.end())
        end = comma if comma >= 0 else end
    out = [m for m in ms if ex.start() <= m[0] < end]
    main = [m for m in ms if m not in out]
    if not main:
        return out[-1:], out[:-1]
    return main, out


def one(m):
    return {m[2]: m[3][0] if m[2] == "corridor" else m[3]}


def exception(out):
    """The exception of a group: names and corridors; a kind counts only if nothing else is there."""
    named = [m for m in out if m[2] != "kind"] or out
    return [one(m) for m in named]


def selector(ms, clause):
    """The selector of one group of mentions, with its exception."""
    keep, drop = split_except(ms, clause)
    sel, union = {}, []
    for m in keep:
        got = one(m)
        if m[2] in sel and sel[m[2]] != got[m[2]]:
            union.append(got)
        else:
            sel.update(got)
    if union:
        sel = {"any_of": [{k: v} for k, v in sel.items()] + union}
    if drop:
        sel["except"] = exception(drop)
    return sel or None


def times(clause):
    """(t1, t2) of a closure; None where the text gives no limit."""
    m = re.search(r"(?:from|between)\s+(?:step\s+)?(\d+)\s+(?:to|and|until|till|through|up to)\s+(?:step\s+)?(\d+)", clause, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"steps\s+(\d+)\s+(?:to|and|through)\s+(\d+)", clause, re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(?:until|till|up to|through|only opens? at|opens? at|reopens? at|before|opens? (?:again )?(?:at|on)|back at)\s+step\s+(\d+)", clause, re.I)
    if m:
        return None, int(m.group(1))
    m = re.search(r"(?:from|after|starting(?: at| from| with)?|beginning(?: at| with)?|as of|once)\s+step\s+(\d+)", clause, re.I)
    if m:
        return int(m.group(1)), None
    return None, None


def side_mentions(ms):
    return [m for m in ms if m[2] in ("customer", "kind", "order", "priority", "station", "corridor")]


class P1:
    def __init__(self, world):
        self.m = Mentions(world)

    def close(self, clause, ms):
        got = [c for m in ms if m[2] == "corridor" for c in m[3]]
        t1, t2 = times(clause)
        rule = {"type": "close", "corridors": sorted(set(got))}
        if t1 is not None:
            rule["t1"] = t1
        if t2 is not None:
            rule["t2"] = t2
        return rule

    def who(self, ms, clause):
        return selector([m for m in ms if m[2] in ("customer", "kind", "order", "priority", "station", "corridor")], clause)

    def apart(self, ms, clause):
        groups, out = split_except(side_mentions(ms), clause)
        if len(groups) < 2:
            return None
        a, b = dict(one(groups[0])), dict(one(groups[1]))
        if re.search(r"\bother customers\b", clause, re.I) and any("customer" in x for x in (a, b)) and any("kind" in x for x in (a, b)):         # X may not share a trip with k orders of other customers
            kind, name = (a, b) if "kind" in a else (b, a)
            kind["except"] = [dict(name)]
            return {"type": "no_mix", "a": kind, "b": name}
        if out:
            named = {m[3] for m in out if m[2] == "kind"}
            host = b if "kind" in b and b["kind"] in named else a
            host["except"] = exception(out)
        return {"type": "no_mix", "a": a, "b": b}

    def pick(self, ms, clause):
        main, out = split_except(side_mentions(ms), clause)
        if not main:
            return None
        low = clause.lower()
        rest = lambda m: {"all": True, "except": [one(m)]}
        if len(main) == 1:
            m = main[0]
            if re.search(r"\b(last|for the end|end with|finish with|nothing else may be picked after|at the end|at the very end|final)\b", low):
                return {"type": "pick_order", "first": rest(m), "then": one(m)}
            if re.search(r"\b(first|start with|begin with|before anything else|ahead of everything|ahead of anything|before everything|before all)\b", low):
                return {"type": "pick_order", "first": one(m), "then": rest(m)}
            return None
        a, b = main[0], main[1]
        between = clause[a[1]:b[0]]
        if re.search(r"\b(never|not|won't|no)\b.*\b(before|ahead of)\b", between, re.I):
            first, second = b, a
        elif re.search(r"\b(before|ahead of|prior to|precede\w*|earlier than|then|followed by)\b", between, re.I):
            first, second = a, b
        elif re.search(r"\b(after|following|later than|once|only when|until|behind)\b", between, re.I):
            first, second = b, a
        elif re.search(r"\b(after|following|once)\b", clause[:a[0]], re.I):
            first, second = a, b
        elif re.search(r"\b(before|ahead of|prior to)\b", clause[:a[0]], re.I):
            first, second = b, a
        else:
            return None
        rule = {"type": "pick_order", "first": dict(one(first)), "then": dict(one(second))}
        if out:
            named = {m[3] for m in out if m[2] == "kind"}
            ex = EXCEPT.search(clause)
            host = "then" if rule["then"].get("kind") in named else "first" if rule["first"].get("kind") in named else None
            host = host or "first"
            rule[host]["except"] = exception(out)
        return rule

    def clause_rules(self, clause):
        if PAST.search(clause) and not MUST.search(clause):
            return []
        ms = self.m.coref(clause, self.m.scan(clause))
        self.m.remember(ms)
        if not ms:
            return []
        kinds = {m[2] for m in ms}
        ex = EXCEPT.search(clause)
        aside = ex and all(m[0] >= ex.start() for m in ms if m[2] == "corridor") and kinds - {"corridor"}
        if "corridor" in kinds and CLOSE.search(clause) and not aside and not re.search(r"\b(pick|picked)\b", clause, re.I):
            return [self.close(clause, ms)]
        if "robot" in kinds:
            robot = next(m[3] for m in ms if m[2] == "robot")
            who = self.who(ms, clause)
            return [{"type": "robot_only", "who": who, "robot": robot}] if who else []
        if APART.search(clause):
            rule = self.apart(ms, clause)
            return [rule] if rule else []
        dl = DELIVER.search(clause) or VAN.search(clause)
        if dl and re.search(r"\bstep\b", clause, re.I) and "corridor" not in kinds:
            who = self.who(ms, clause)
            return [{"type": "deadline", "who": who, "t": int(dl.group(1))}] if who else []
        if "station" in kinds and (STATION_WORD.search(clause) or re.search(r"\bstation [A-F] takes\b", clause)):
            st = [m for m in ms if m[2] == "station"]
            rest = [m for m in ms if m[2] != "station"]
            who = selector(rest, clause) if rest else None
            return [{"type": "station", "who": who, "station": st[0][3]}] if who else []
        if PICK.search(clause):
            rule = self.pick(ms, clause)
            return [rule] if rule else []
        return []


def clauses(sentence, m):
    pieces, out = SPLIT.split(sentence), []
    for p in pieces:
        if out and not m.scan(p) and not re.search(r"\b(they|them|those|these|it|their)\b", p, re.I):
            out[-1] += ", " + p
        elif out and not m.scan(out[-1]):
            out[-1] += ", " + p
        else:
            out.append(p)
    return out


def restated(rules):
    seen, out = set(), []
    for r in rules:
        key = repr(sorted((k, repr(v)) for k, v in r.items()))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def parse_p1(text, world):
    p = P1(world)
    rules = []
    for s in sentences(text):
        for c in clauses(s, p.m):
            rules += p.clause_rules(c)
    return {"rules": restated(rules)}


def parse_p0(text, world):
    """One rule per sentence: the first keyword decides, with the first mentions."""
    m = Mentions(world)
    rules = []
    for s in sentences(text):
        ms = m.scan(s)
        who = next(({x[2]: x[3]} for x in ms if x[2] in ("customer", "kind", "order", "priority")), None)
        corridors = [c for x in ms if x[2] == "corridor" for c in x[3]]
        step = re.search(r"step\s+(\d+)", s, re.I)
        robot = next((x[3] for x in ms if x[2] == "robot"), None)
        station = next((x[3] for x in ms if x[2] == "station"), None)
        kinds = [x for x in ms if x[2] == "kind"]
        low = s.lower()
        if corridors and re.search(r"closed|out of use|keep|nobody|no robot|opens", low):
            nums = [int(x) for x in re.findall(r"step\s+(\d+)", s, re.I)]
            rule = {"type": "close", "corridors": corridors}
            if len(nums) >= 2:
                rule.update(t1=nums[0], t2=nums[1])
            rules.append(rule)
        elif robot and who:
            rules.append({"type": "robot_only", "who": who, "robot": robot})
        elif who and re.search(r"trip|together|mix", low) and len(ms) >= 2 and "pick" not in low:
            other = next(({x[2]: x[3]} for x in ms[1:] if {x[2]: x[3]} != who), None)
            if other:
                rules.append({"type": "no_mix", "a": who, "b": other})
        elif who and step:
            rules.append({"type": "deadline", "who": who, "t": int(step.group(1))})
        elif who and station:
            rules.append({"type": "station", "who": who, "station": station})
        elif len(kinds) >= 2 and "before" in low:
            rules.append({"type": "pick_order", "first": {"kind": kinds[0][3]}, "then": {"kind": kinds[1][3]}})
        elif len(kinds) >= 2 and "after" in low:
            rules.append({"type": "pick_order", "first": {"kind": kinds[1][3]}, "then": {"kind": kinds[0][3]}})
        elif len(kinds) == 1 and "last" in low:
            rules.append({"type": "pick_order", "first": {"all": True, "except": [{"kind": kinds[0][3]}]}, "then": {"kind": kinds[0][3]}})
        elif len(kinds) == 1 and "first" in low:
            rules.append({"type": "pick_order", "first": {"kind": kinds[0][3]}, "then": {"all": True, "except": [{"kind": kinds[0][3]}]}})
    return {"rules": rules}


def is_constraint(clause):
    if PAST.search(clause) and not MUST.search(clause):
        return False
    return bool(WORDS.search(clause) or CLOSE.search(clause) or APART.search(clause))


def send_to_llm(text, world):
    """True if p1 leaves a clause with constraint words uncovered, or gives fewer valid rules than there are constraint sentences."""
    ctx = L.Context(world)
    p = P1(world)
    needs = uncovered = 0
    for s in sentences(text):
        got, words = [], False
        for c in clauses(s, p.m):
            found = p.clause_rules(c)
            got += found
            if is_constraint(c):
                words = True
                uncovered += not found
        if words:
            needs += 1
            uncovered += not L.validate({"rules": got}, ctx)[0]
    return uncovered > 0 or len(L.validate(parse_p1(text, world), ctx)[0]) < needs
