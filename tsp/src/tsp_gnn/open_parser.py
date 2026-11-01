"""Arm p1: a rule-based parser of open-language rules into the DSL (no LLM); the fair non-LLM baseline of step 3.

Written in a limited effort (about 2 h) from the L1 sentence templates (tsp_gnn.open_gen) and the held-out development
paragraphs of L2 (instances the test set does not use). It was frozen before the L2 audit. Steps:
    1. the text is cut into sentences (the fixed preamble is dropped) and each sentence into clauses at semicolons and
       at ", and" / ", but" / ", while" when both sides have a place or a group;
    2. each clause is scanned for mentions: place names (legend), customer numbers, "the k listed places / <types>
       closest to X", "<type> in <area>", "<type>", "<area>", "every place in <area>", "the closest <type>";
    3. "except / other than / apart from / unless ... X" attaches the named place to the group that contains it;
    4. the type of the rule comes from keywords, in this order: times (window) -> zone words (soft zone) -> adjacency
       words (forbidden if a negation word is present, mandatory otherwise) -> order words (precedence);
    5. the direction of a precedence comes from the order word and the position of the two sides: "A before B",
       "B after A", "no B until A", "do not visit B until A";
    6. coreference: "the latter", "the same place", "that <type>", "it" point to the last place mentioned;
    7. sentences that say nothing about order, adjacency, time or zones give no rule; remarks about the past
       ("last week", "was", "asked for", ...) are dropped;
    8. tuned on the dev paragraphs: spelling is normalised first (times with a colon, "residences" for houses) and the
       writer's stock filler "it is not acceptable to skip ..." is removed; restatements of a rule with the same two
       sides are dropped, keeping the first one that carries an exception.
"""

import re

from tsp_gnn import keyword_parser as K
from tsp_gnn import open_dsl as D
from tsp_gnn import rules as R

PLURAL = {"hospitals": "hospital", "schools": "school", "warehouses": "warehouse", "shops": "shop", "houses": "house",
          "pharmacies": "pharmacy", "offices": "office"}
KIND_WORD = "|".join(list(PLURAL) + list(PLURAL.values()))
PLACE_WORD = r"(?:listed )?(?:place|location|stop|site)s?"
TIME = re.compile(r"(?<![\w.])\d+\.\d+")
ADJ = re.compile(r"\b(direct(?:ly)?|straight|consecutive(?:ly)?|next to|adjacent|one right after the other|"
                 r"back[- ]to[- ]back|immediately|linked?|legs?|right after|beside|connection|separated|apart(?! from))\b", re.I)
NEG = re.compile(r"\b(never|not|no|cannot|can't|closed|forbidden|prohibited|unavailable|avoid|mustn't|don't|isn't|"
                 r"nor|without|refrain|unacceptable)\b", re.I)
BEFORE = re.compile(r"\b(before|prior to|ahead of|earlier than|precede[sd]?|first|until|preceding)\b", re.I)
AFTER = re.compile(r"\b(after|later than|following|follows?|once|only when|since|subsequent)\b", re.I)
UNTIL = re.compile(r"\b(until|unless|till|before)\b", re.I)
EXCEPT = re.compile(r"\b(except(?: for)?|other than|apart from|aside from|excluding|but not|save for|besides|"
                    r"with the (?:sole |only |single )?exception of)\b", re.I)
SOFT = re.compile(r"\b(avoid|minimi[sz]e|minimum|stay out|keep|limit|discouraged|steer clear|as little|reduce)\b", re.I)
PAST = re.compile(r"\b(last (?:week|month|year)|yesterday|recently|was|were|has a new|asked for|ordered|belong|"
                  r"repainted|painted|pay|paying|payment|fuel|radio|complained|coincidence|new to|requested|"
                  r"packaging|receipt|manager|owned|ownership|busy|fluke)\b", re.I)
RULE_WORD = re.compile(r"\b(must|may|can|cannot|should|has to|have to|need|needs|do not|don't|never|no |only|ensure|"
                       r"make sure|visit|serve|go|drive|avoid|keep|open|comes?|closed|linked|essential|head|proceed|"
                       r"prioriti[sz]e|position|includes?)\b", re.I)
COREF = re.compile(r"\b(the latter|the former|the same place|that (?:place|site|location|one)|this place|"
                   r"th(?:at|is) (?:" + KIND_WORD + r")|it(?!['\u2019]s| is| was| will| has| would| can| should| may))\b", re.I)


FILLER = re.compile(r"[,;]?\s*(?:and\s+)?\bit(?:'s|\u2019s| is) not acceptable to skip\b[^.;]*(?:[.;]|$)", re.I)


def normalise(text):
    """Spelling of the writer of L2: colons in times, other words for a house, its stock double negation."""
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1.\2", text)
    text = re.sub(r"\bresidential (?:deliver(?:y|ies)|addresses|stops?|homes?)\b|\bresidences?\b", "houses", text,
                  flags=re.I)
    return FILLER.sub(".", text)


def split_sentences(text):
    out = []
    for line in normalise(text).splitlines():
        if line.startswith("Route rules.") or line.startswith("Named places:"):
            continue
        line = line[2:] if line.startswith("- ") else line
        # protect decimals, then cut at sentence ends
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", line.strip())
        out += [p.strip() for p in parts if p.strip()]
    return out


def split_clauses(sentence, has_ref):
    pieces, out = re.split(r"\s*;\s*|\s+[\u2013\u2014-]\s+|\s*[,]\s*(?:and|but|while|whereas|plus|so|as)\s+", sentence), []
    for p in pieces:
        if out and not has_ref(p):                     # a tail without a place belongs to the clause before it
            out[-1] += ", " + p
        elif out and not has_ref(out[-1]):
            out[-1] += ", " + p
        else:
            out.append(p)
    return out


class Parser:
    def __init__(self, legend, n):
        self.legend, self.n = legend, n
        self.names = sorted((p["name"] for p in legend if p["name"]), key=len, reverse=True)
        self.areas = sorted({p["area"] for p in legend if p["area"]})
        self.name_re = re.compile("|".join(re.escape(x) for x in self.names), re.I) if self.names else None
        self.area_re = re.compile(r"\b(" + "|".join(self.areas) + r")\b", re.I) if self.areas else None
        self.last = None

    def proper(self, x):
        return next(nm for nm in self.names if nm.lower() == x.lower())

    def mentions(self, clause):
        """Selectors in the clause, in order: [(start, end, selector)]. The longest reading of each span wins."""
        found = []
        taken = []

        def add(s, e, sel):
            if not any(s < b and a < e for a, b in taken):
                taken.append((s, e))
                found.append((s, e, sel))
        name = self.name_re.pattern if self.name_re else r"(?!x)x"
        area = self.area_re.pattern if self.area_re else r"(?!x)x"
        count = {"two": 2, "three": 3, "four": 4, "five": 5}
        for m in re.finditer(rf"\b(\d+|two|three|four|five)\s+(?:(?:closest|nearest)\s+)?({PLACE_WORD}|{KIND_WORD})\s+"
                             rf"(?:that are |which are )?(?:closest\s+|nearest\s+)?to\s+({name})", clause, re.I):
            k = count.get(m.group(1).lower(), None) or int(m.group(1))
            sel = {"nearest": {"k": k, "of": self.proper(m.group(3))}}
            word = m.group(2).lower()
            if word in PLURAL or word in PLURAL.values():
                sel["type_is"] = PLURAL.get(word, word)
            add(m.start(), m.end(), sel)
        for m in re.finditer(rf"(?:closest|nearest)\s+({PLACE_WORD}|{KIND_WORD})(?:\s+to\s+({name}|it))?", clause,
                             re.I):
            of = self.proper(m.group(2)) if m.group(2) and m.group(2).lower() != "it" else None
            if of is None:
                nm = list(self.name_re.finditer(clause[:m.start()])) if self.name_re else []
                of = self.proper(nm[-1].group()) if nm else None
            if of is None:
                continue
            sel = {"nearest": {"k": 1, "of": of}}
            word = m.group(1).lower()
            if word in PLURAL or word in PLURAL.values():
                sel["type_is"] = PLURAL.get(word, word)
            add(m.start(), m.end(), sel)
        for m in re.finditer(rf"(?:listed )?place (?:closest|nearest) to ({name}|it)", clause, re.I):
            nm = list(self.name_re.finditer(clause[:m.start()])) if self.name_re else []
            of = self.proper(nm[-1].group()) if m.group(1).lower() == "it" and nm else m.group(1)
            if of.lower() != "it":
                add(m.start(), m.end(), {"nearest": {"k": 1, "of": self.proper(of)}})
        for m in re.finditer(rf"\b({KIND_WORD})\s+(?:located\s+|situated\s+)?in\s+({area})\b", clause, re.I):
            word = m.group(1).lower()
            add(m.start(), m.end(), {"type_is": PLURAL.get(word, word),
                                     "area_is": next(a for a in self.areas if a.lower() == m.group(2).lower())})
        for m in re.finditer(rf"\b(?:any|every|all|each)?\s*(?:place|site|location|stop)s?\s+in\s+({area})\b", clause,
                             re.I):
            add(m.start(), m.end(), {"area_is": next(a for a in self.areas if a.lower() == m.group(1).lower())})
        for m in re.finditer(rf"(?:\b{area})\s+({KIND_WORD})\b", clause, re.I):
            word = m.groups()[-1].lower()
            place = m.group().rsplit(None, 1)[0].lower()
            add(m.start(), m.end(), {"type_is": PLURAL.get(word, word),
                                     "area_is": next(a for a in self.areas if a.lower() == place)})
        if self.name_re:
            for m in self.name_re.finditer(clause):
                add(m.start(), m.end(), self.proper(m.group()))
        for m in re.finditer(rf"\b({KIND_WORD})\b", clause, re.I):
            word = m.group(1).lower()
            if COREF.fullmatch(clause[max(m.start() - 5, 0):m.end()].strip()):
                continue
            add(m.start(), m.end(), {"type_is": PLURAL.get(word, word)})
        for m in re.finditer(r"\b(?:customers?|stops?|clients?)\s+(\d+)\b", clause, re.I):
            if 1 <= int(m.group(1)) < self.n:
                add(m.start(), m.end(), int(m.group(1)))
        found.sort(key=lambda x: x[0])
        return found

    def coref(self, clause, ms):
        if self.last is None:
            return ms
        taken = [(a, b) for a, b, _ in ms]
        out = list(ms)
        for m in COREF.finditer(clause):
            if not any(m.start() < b and a < m.end() for a, b in taken):
                out.append((m.start(), m.end(), self.last))
        return sorted(out, key=lambda x: x[0])

    def exceptions(self, clause, ms):
        """(mentions without the excepted places, excepted) where excepted is attached to the group mention nearest
        before it (or the next group if none)."""
        out = [list(m) for m in ms]
        for ex in EXCEPT.finditer(clause):
            after = [m for m in out if m[0] >= ex.end() and isinstance(m[2], (str, int))]
            if not after:
                continue
            target = after[0]
            groups = [m for m in out if isinstance(m[2], dict) and "nearest" not in m[2].get("except", {})]
            prior = [m for m in groups if m[1] <= ex.start()]
            nxt = [m for m in groups if m[0] >= target[1]]
            host = prior[-1] if prior else (nxt[0] if nxt else None)
            if host is None:
                continue
            host[2] = {**host[2], "except": [target[2]]}
            out.remove(target)
        return [tuple(m) for m in out]

    def parse_clause(self, clause):
        rules = []
        for z in dict.fromkeys(m.group(1).lower() for m in K.ZONE_WORD.finditer(clause)):
            if re.search(r"\bzones?\b", clause, re.I) and (SOFT.search(clause) or NEG.search(clause)):
                rules.append({"type": "soft_zone", "zone": z})
        ms = self.exceptions(clause, self.coref(clause, self.mentions(clause)))
        for _, _, s in ms:
            if isinstance(s, str):
                self.last = s
        if not ms:
            return rules
        times = [float(x) for x in TIME.findall(clause)]
        if len(times) >= 2:
            areas = {m[2]["type_is"] for m in ms if isinstance(m[2], dict) and "area_is" in m[2] and "type_is" in m[2]}
            ms = [m for m in ms if not (isinstance(m[2], dict) and m[2].keys() == {"type_is"} and m[2]["type_is"] in areas)]
            who = ms[0][2] if len(ms) == 1 else {"any_of": [m[2] for m in ms]}
            rules.append({"type": "time_window", "who": who, "t1": min(times[:2]), "t2": max(times[:2])})
            return rules
        if len(ms) < 2:
            return rules
        a, b = ms[0], ms[1]
        if len(ms) > 2:                                 # "both A and B ..." on one side: join neighbours around "and"
            joined = []
            for m in ms:
                if joined and re.fullmatch(r"\s*(,|and|or|,\s*and)\s*", clause[joined[-1][1]:m[0]]):
                    prev = joined[-1]
                    sel = prev[2].get("any_of") if isinstance(prev[2], dict) and "any_of" in prev[2] else [prev[2]]
                    joined[-1] = (prev[0], m[1], {"any_of": sel + [m[2]]})
                else:
                    joined.append(m)
            if len(joined) >= 2:
                a, b = joined[0], joined[1]
            words = [w for rx in (BEFORE, AFTER) for w in rx.finditer(clause)]
            pos = max((w.start() for w in words), default=None)
            left = [m for m in joined if pos is not None and m[1] <= pos]
            right = [m for m in joined if pos is not None and m[0] >= pos]
            if len(joined) > 2 and left and right:     # three sides: the two around the last order word
                a, b = left[-1], right[0]
        span = clause
        adj = ADJ.search(span)
        before, after = BEFORE.search(span), AFTER.search(span)
        if adj and not (before and not re.search(r"direct|straight|next|consecutive|adjacent|link|leg", span, re.I)):
            kind = "forbidden_edge" if NEG.search(span) else "mandatory_edge"
            rules.append({"type": kind, "a": a[2], "b": b[2]})
            return rules
        if not (before or after):
            return rules
        # "No B until A" / "do not visit B until A": the first side comes second on the route
        if re.search(r"\b(no|not|never|cannot|can't|don't|mustn't|nor)\b", clause[:a[0] + 1], re.I) and \
                UNTIL.search(clause[a[1]:b[0]]):
            first, second = b, a
        elif before and before.start() < a[0] and re.search(r"\bfirst\b", before.group(), re.I):
            first, second = a, b
        else:
            between = clause[a[1]:b[0]]
            if BEFORE.search(between):
                first, second = a, b
            elif AFTER.search(between):
                first, second = b, a
            elif after and after.start() < a[0]:        # "After A, B ..."
                first, second = a, b
            elif before and before.start() < a[0]:      # "Before A, B ..."
                first, second = b, a
            else:
                first, second = a, b
        rules.append({"type": "precedence", "before": first[2], "after": second[2]})
        return rules

    def sentence_rules(self, sentence):
        if not RULE_WORD.search(sentence) and not K.ZONE_WORD.search(sentence):
            return []
        out = []
        for clause in split_clauses(sentence, lambda c: bool(self.mentions(c)) or bool(K.ZONE_WORD.search(c))):
            if PAST.search(clause) and not re.search(r"\b(must|may not|never|cannot|has to|have to|zones?)\b", clause, re.I):
                continue
            out += self.parse_clause(clause)
        return out


def bare(sel):
    """A selector as a hashable value, without its exception."""
    if isinstance(sel, dict):
        return tuple(sorted((k, repr(v)) for k, v in sel.items() if k != "except"))
    return sel


def side_key(r):
    """Two rules on the same two sides restate each other (an edge rule is the same with either sign and order)."""
    if r["type"] == "precedence":
        return ("precedence", repr(bare(r["before"])), repr(bare(r["after"])))
    if r["type"] in ("forbidden_edge", "mandatory_edge"):
        return ("edge", frozenset((repr(bare(r["a"])), repr(bare(r["b"])))))
    return None


def drop_restatements(rules):
    out, seen = [], {}
    for r in rules:
        key = side_key(r)
        if key is None or key not in seen:
            if key is not None:
                seen[key] = len(out)
            out.append(r)
        elif "except" in repr(r) and "except" not in repr(out[seen[key]]):
            out[seen[key]] = r
    return out


def parse(text, legend, n):
    """Arm p1: the DSL answer {"rules": [...]} for a text."""
    p = Parser(legend, n)
    rules = []
    for s in split_sentences(text):
        rules += p.sentence_rules(s)
    return {"rules": drop_restatements(rules)}


def parse_l0(text, n):
    """L0 keeps the step-2 style (numbers, "Named places:" line), which the b0+ parser already reads: p1 uses it."""
    from tsp_gnn.keyword_parser import parse_keywords
    prims = parse_keywords(text, n)
    conv = {"precedence": lambda f: {"before": f["before"], "after": f["after"]},
            "forbidden_edge": lambda f: {"a": f["i"], "b": f["j"]}, "mandatory_edge": lambda f: {"a": f["i"], "b": f["j"]},
            "time_window": lambda f: {"who": f["customer"], "t1": f["t1"], "t2": f["t2"]},
            "soft_zone": lambda f: {"zone": f["zone"]}}
    return {"rules": [{"type": r["type"], **conv[r["type"]](r["formal"])} for r in prims]}


def p0_text(legend, text):
    """The old keyword parser reads bullet lines and a "Named places:" line: the instance is written that way, one
    bullet per sentence, so that p0 gets its best chance on L1 and L2."""
    names = "Named places: " + "; ".join(f"customer {p['id']} is {p['name']}" for p in legend) + "."
    return "\n".join([R.PREAMBLE, names] + [f"- {s}" for s in split_sentences(text)])


assert D.KINDS                                     # the type words above are the DSL's
