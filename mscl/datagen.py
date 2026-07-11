"""MSCL v1 — synthetic data generator (Piece 4).

Generates (english, object_table, gold_spec_json) triples for training/evaluating the
English->MSCL model. Two streams:
  * UNAMBIGUOUS : random satisfiable specs over the 17 types + relations.
  * AMBIGUOUS   : specs whose English deliberately underspecifies something, so the gold
                  contains a CHOICE node (direction / offset / reference / unsupported_type
                  / scope). This is what teaches the model to EMIT CHOICE.

English is produced by TEMPLATES (deterministic, runs here). A `paraphrase_hook` is
provided so you can later rewrite the robotic templates with your local model (you run
that step); the gold logic is unchanged by paraphrasing.

Every generated spec is checked satisfiable via the feasibility oracle before being kept,
so the dataset never contains impossible unambiguous specs.
"""
from __future__ import annotations
import random, json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
from .ast import (Spec, Obj, Relation, TypePred, PropertyPred, Default,
                  And, Or, Not, Choice, Option, Formula)
from .json_io import spec_to_json
from .feasibility import collect_atoms, init_domains, feasible
from .validate import validate
from . import profile

# ---------------------------------------------------------------------------
# vocabulary for verbalization
# ---------------------------------------------------------------------------
TYPES = list(profile.SPRING_TYPES)
COLORS = ["red", "blue", "green", "black", "white", "brown", "wooden", "metal"]
# relation -> (english phrase template with {a} {b} and optional {c})
REL_PHRASE = {
    "cright":   "{a} completely to the right of {b}",
    "cleft":    "{a} completely to the left of {b}",
    "cabove":   "{a} completely above {b}",
    "cbelow":   "{a} completely below {b}",
    "right":    "{a} to the right of {b}",
    "left":     "{a} to the left of {b}",
    "above":    "{a} above {b}",
    "below":    "{a} below {b}",
    "wider":    "{a} wider than {b}",
    "narrower": "{a} narrower than {b}",
    "taller":   "{a} taller than {b}",
    "shorter":  "{a} shorter than {b}",
    "xeq":      "{a} vertically aligned with {b}",
    "yeq":      "{a} horizontally aligned with {b}",
    "weq":      "{a} the same width as {b}",
    "heq":      "{a} the same height as {b}",
}
VALUE_PHRASE = {
    "right_value":  "{a} in the right half of the image",
    "left_value":   "{a} in the left half of the image",
    "above_value":  "{a} in the top half of the image",
    "below_value":  "{a} in the bottom half of the image",
    "wider_value":  "{a} wider than {c} per-mille",
    "taller_value": "{a} taller than {c} per-mille",
}
# directional relations that admit an offset and can be made AMBIGUOUS about magnitude
OFFSETABLE = ["cright", "cleft", "cabove", "cbelow", "right", "left", "above", "below"]
# words that introduce a vague direction (-> direction CHOICE)
VAGUE_DIR = ["by", "near", "next to", "beside"]

ParaphraseHook = Optional[Callable[[str], str]]


@dataclass
class Sample:
    english: str
    objects_json: List[dict]
    gold_json: dict
    ambiguous: bool
    meta: dict


def _art(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _noun(o: Obj) -> str:
    props = " ".join(o.properties)
    t = o.type if o.type else "object"
    base = f"{props} {t}".strip()
    return f"{_art(base)} {base}"


# ---------------------------------------------------------------------------
# spec sampling
# ---------------------------------------------------------------------------
def _sample_objects(rng: random.Random, n_new: int, n_existing: int,
                    *, unique_types: bool = False) -> List[Obj]:
    chosen_types = rng.sample(TYPES, n_new + n_existing) if unique_types else None
    ti = 0
    objs = []
    for i in range(n_existing):
        t = chosen_types[ti] if chosen_types else rng.choice(TYPES)
        ti += 1
        x, y = rng.randint(0, 700), rng.randint(0, 700)
        w, h = rng.randint(80, 300), rng.randint(80, 300)
        objs.append(Obj(f"e{i}", "existing", t, box=(x, y, w, h)))
    for i in range(n_new):
        t = chosen_types[ti] if chosen_types else rng.choice(TYPES)
        ti += 1
        props = []
        if rng.random() < 0.6:
            props = [rng.choice(COLORS)]
        objs.append(Obj(f"o{i}", "new", t, properties=props))
    return objs


def _try_make_unambiguous(rng: random.Random) -> Optional[Tuple[Spec, List[str]]]:
    n_new = rng.randint(1, 3)
    n_existing = rng.randint(0, 2)
    # Duplicate types made supposedly unambiguous sentences impossible to bind to ids
    # (for example, "a mirror ... a mirror").  Reserve duplicates for reference-CHOICE data.
    objs = _sample_objects(rng, n_new, n_existing, unique_types=True)
    ids = [o.id for o in objs]
    new_ids = [o.id for o in objs if o.status == "new"]

    conj: List[Formula] = []
    phrases: List[str] = []
    nounmap = {o.id: _noun(o) for o in objs}

    for o in objs:
        if o.status == "new":
            conj.append(TypePred(o.id, o.type))
            for p in o.properties:
                conj.append(PropertyPred(o.id, p))
            conj.append(Default(o.id))

    n_constraints = rng.randint(1, min(4, max(1, len(ids))))
    used = set()
    for _ in range(n_constraints):
        if len(ids) >= 2 and rng.random() < 0.8:
            a, b = rng.sample(ids, 2)
            if a not in new_ids and b not in new_ids:
                continue  # constraint should involve at least one new object
            name = rng.choice(list(REL_PHRASE))
            key = (name, a, b)
            if key in used:
                continue
            used.add(key)
            c = rng.choice([None, None, rng.randint(50, 300)]) if name in OFFSETABLE else None
            conj.append(Relation(name, [a, b], c))
            ph = REL_PHRASE[name].format(a=nounmap[a], b=nounmap[b])
            if c:
                ph += f" by {c} per-mille"
            phrases.append(ph)
        else:
            a = rng.choice(new_ids)
            name = rng.choice(list(VALUE_PHRASE))
            c = rng.randint(200, 600) if "value" in name and name.endswith("value") else None
            if name in ("wider_value", "taller_value"):
                c = rng.randint(200, 500)
                conj.append(Relation(name, [a], c))
                phrases.append(VALUE_PHRASE[name].format(a=nounmap[a], c=c))
            else:
                # "right/left/top/bottom half" has a learnable, explicit midpoint threshold.
                conj.append(Relation(name, [a], 500))
                phrases.append(VALUE_PHRASE[name].format(a=nounmap[a]))

    spec = Spec(objs, And(conj) if len(conj) > 1 else conj[0])
    # satisfiability check (existing objects pinned)
    existing = {o.id: o.box for o in objs if o.box}
    atoms = collect_atoms(spec.formula)
    if atoms is None:
        return None
    dom = init_domains(ids, existing)
    if not feasible(atoms, dom):
        return None
    return spec, phrases


def sample_unambiguous(rng: random.Random, paraphrase: ParaphraseHook = None) -> Optional[Sample]:
    out = _try_make_unambiguous(rng)
    if out is None:
        return None
    spec, phrases = out
    objs = spec.objects
    intro = ", ".join(p for p in phrases)
    english = f"Add {', '.join(_noun(o) for o in objs if o.status=='new')}, " \
              f"with {intro}." if phrases else \
              f"Add {', '.join(_noun(o) for o in objs if o.status=='new')}."
    if paraphrase:
        english = paraphrase(english)
    validate(spec)
    return Sample(english, [json.loads(json.dumps(o, default=lambda x: None)) for o in []],
                  gold_json={}, ambiguous=False, meta={})  # filled below


# The above sample_* returns are normalized through this builder to keep JSON consistent:
def _finish(spec: Spec, english: str, ambiguous: bool, meta: dict,
            paraphrase: ParaphraseHook) -> Sample:
    if paraphrase:
        english = paraphrase(english)
    j = spec_to_json(spec)
    return Sample(english=english, objects_json=j["objects"],
                  gold_json=j, ambiguous=ambiguous, meta=meta)


def gen_unambiguous(rng: random.Random, paraphrase: ParaphraseHook = None) -> Optional[Sample]:
    out = _try_make_unambiguous(rng)
    if out is None:
        return None
    spec, phrases = out
    new_nouns = ", ".join(_noun(o) for o in spec.objects if o.status == "new")
    english = (f"Add {new_nouns}, with " + ", ".join(phrases) + ".") if phrases else f"Add {new_nouns}."
    return _finish(spec, english, False, {"kind": "unambiguous", "n_constraints": len(phrases)}, paraphrase)


# ---------------------------------------------------------------------------
# ambiguous generation — each kind produces a gold CHOICE
# ---------------------------------------------------------------------------
def gen_ambiguous_direction(rng: random.Random, paraphrase: ParaphraseHook = None) -> Optional[Sample]:
    """'put X by Y' -> direction CHOICE over {cleft,cright,cabove,cbelow}."""
    objs = _sample_objects(rng, 1, 1)
    new, ex = objs[1], objs[0]   # _sample_objects puts existing first
    # ensure ordering: existing then new
    ex = [o for o in objs if o.status == "existing"][0]
    new = [o for o in objs if o.status == "new"][0]
    dirs = ["cleft", "cright", "cabove", "cbelow"]
    rng.shuffle(dirs)
    vague = rng.choice(VAGUE_DIR)
    opts = [Option(round(p, 2), Relation(d, [new.id, ex.id]))
            for d, p in zip(dirs, _priors(rng, len(dirs)))]
    choice = Choice("direction", f"{vague} the {ex.type}", opts)
    conj = [TypePred(new.id, new.type), Default(new.id), choice]
    for p in new.properties:
        conj.append(PropertyPred(new.id, p))
    spec = Spec(objs, And(conj))
    english = f"Put {_noun(new)} {vague} the {ex.type}."
    return _finish(spec, english, True, {"kind": "ambiguous_direction"}, paraphrase)


def gen_ambiguous_offset(rng: random.Random, paraphrase: ParaphraseHook = None) -> Optional[Sample]:
    """'X well to the left of Y' (emphasis) -> offset CHOICE {c=0, c=big}, emphasis=True."""
    objs = _sample_objects(rng, 1, 1)
    ex = [o for o in objs if o.status == "existing"][0]
    new = [o for o in objs if o.status == "new"][0]
    d = rng.choice(["cleft", "cright", "cabove", "cbelow"])
    big = rng.choice([150, 200, 300])
    opts = [Option(0.4, Relation(d, [new.id, ex.id], 0)),
            Option(0.6, Relation(d, [new.id, ex.id], big))]
    choice = Choice("offset", "well", opts, emphasis=True)
    conj = [TypePred(new.id, new.type), Default(new.id), choice]
    for p in new.properties:                      # gold must match the English's properties
        conj.append(PropertyPred(new.id, p))
    spec = Spec(objs, And(conj))
    dirword = {"cleft": "well to the left of", "cright": "well to the right of",
               "cabove": "well above", "cbelow": "well below"}[d]
    english = f"Put {_noun(new)} {dirword} the {ex.type}."
    return _finish(spec, english, True, {"kind": "ambiguous_offset"}, paraphrase)


def gen_ambiguous_reference(rng: random.Random, paraphrase: ParaphraseHook = None) -> Optional[Sample]:
    """Two existing objects of the SAME type; 'the X' -> reference CHOICE over the two ids."""
    t = rng.choice(TYPES)
    e0 = Obj("e0", "existing", t, box=(rng.randint(0, 300), rng.randint(0, 700),
                                       rng.randint(80, 200), rng.randint(80, 200)))
    e1 = Obj("e1", "existing", t, box=(rng.randint(500, 800), rng.randint(0, 700),
                                       rng.randint(80, 200), rng.randint(80, 200)))
    new = Obj("o0", "new", rng.choice([x for x in TYPES if x != t]))
    d = rng.choice(["left", "right"])
    opts = [Option(0.5, Relation(d, [new.id, "e0"])),
            Option(0.5, Relation(d, [new.id, "e1"]))]
    choice = Choice("reference", f"the {t}", opts)
    spec = Spec([e0, e1, new], And([TypePred(new.id, new.type), Default(new.id), choice]))
    side = "left" if d == "left" else "right"
    english = f"Add {_noun(new)} to the {side} of the {t}."
    return _finish(spec, english, True, {"kind": "ambiguous_reference"}, paraphrase)


def gen_ambiguous_unsupported(rng: random.Random, paraphrase: ParaphraseHook = None) -> Optional[Sample]:
    """Ask for an out-of-vocab object -> unsupported_type CHOICE {nearest types..., SKIP}."""
    word = rng.choice(["lamp", "rug", "clock", "vase", "bookshelf", "fan", "painting"])
    ex = _sample_objects(rng, 0, 1)[0]
    new = Obj("L", "new", None)
    nearest = profile.nearest_types(word, k=2)
    opts = [Option(0.5, TypePred("L", nearest[0])),
            Option(0.3, TypePred("L", nearest[1])),
            Option(0.2, None, skip=True)]
    utype = Choice("unsupported_type", word, opts)
    d = rng.choice(["cleft", "cright", "cabove", "cbelow"])
    dirchoice = Choice("direction", f"near the {ex.type}",
                       [Option(round(p, 2), Relation(dd, ["L", ex.id]))
                        for dd, p in zip(["cleft", "cright", "cabove", "cbelow"], _priors(rng, 4))])
    spec = Spec([ex, new], And([utype, Default("L"), dirchoice]))
    english = f"Put {_art(word)} {word} near the {ex.type}."
    return _finish(spec, english, True,
                   {"kind": "ambiguous_unsupported", "word": word}, paraphrase)


def _priors(rng: random.Random, n: int) -> List[float]:
    xs = [rng.random() + 0.1 for _ in range(n)]
    s = sum(xs)
    return [x / s for x in xs]


def _spec_has_feasible_path(spec: Spec) -> bool:
    """True if at least one combination of CHOICE options is jointly feasible.
    Cheap check: each CHOICE independently has >=1 feasible option against the hard skeleton."""
    from .dialogue import _all_choices, _option_feasible
    existing = {o.id: o.box for o in spec.objects if o.box}
    for ch in _all_choices(spec.formula):
        if not any(_option_feasible(spec, ch, o, existing) for o in ch.options):
            return False
    return True


AMBIG_GENERATORS = [gen_ambiguous_direction, gen_ambiguous_offset,
                    gen_ambiguous_reference, gen_ambiguous_unsupported]


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------
def generate_dataset(n: int, ambiguous_frac: float = 0.5, seed: int = 0,
                     paraphrase: ParaphraseHook = None,
                     max_tries_mult: int = 20) -> List[Sample]:
    rng = random.Random(seed)
    out: List[Sample] = []
    tries = 0
    while len(out) < n and tries < n * max_tries_mult:
        tries += 1
        if rng.random() < ambiguous_frac:
            g = rng.choice(AMBIG_GENERATORS)
            s = g(rng, paraphrase)
        else:
            s = gen_unambiguous(rng, paraphrase)
        if s is not None:
            try:
                # final well-formedness guard
                from .json_io import spec_from_json
                sp = spec_from_json(s.gold_json)
                validate(sp)
                if s.ambiguous and not _spec_has_feasible_path(sp):
                    continue   # reject impossible ambiguous specs
                out.append(s)
            except Exception:
                continue
    return out


def to_jsonl(samples: List[Sample]) -> str:
    lines = []
    for s in samples:
        lines.append(json.dumps({
            "english": s.english,
            "objects": s.objects_json,
            "formula": s.gold_json["formula"],
            "ambiguous": s.ambiguous,
            "meta": s.meta,
        }))
    return "\n".join(lines)
