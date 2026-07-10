"""MSCL v1 — Piece 3: evaluation harness.

Three evaluations, all GPU-free (the model call is abstracted behind a backend):

  1. parse_accuracy(samples, backend)
       exact-match + semantic-equivalence of parsed spec vs gold.
  2. roundtrip_consistency(n, backend)
       gold spec -> english (already paired) -> parse -> equivalence.  Self-supervised,
       unlimited data.
  3. policy_eval(samples)
       on AMBIGUOUS gold specs, run the dialogue policy with a gold-driven simulated user
       and compare {questions asked, final correctness} against ASK_ALL / ASK_NONE / ASK_RANDOM.
"""
from __future__ import annotations
import json, random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
from .ast import Spec, Choice, Option, Relation, TypePred, And, Or, Not, Formula
from .json_io import spec_from_json, spec_to_json, node_to_json
from .parser import parse
from .dialogue import resolve, resolve_ask_none, resolve_ask_all, _all_choices, Question
from .render import to_spring
from .feasibility import collect_atoms, init_domains, feasible


# ---------------------------------------------------------------------------
# equivalence
# ---------------------------------------------------------------------------
def _canon(formula_json: dict) -> str:
    """Canonical string of a CHOICE-free formula for exact comparison: sort AND/OR args."""
    def rec(n):
        if n["node"] in ("and", "or"):
            kids = sorted((rec(a) for a in n["args"]))
            return f"{n['node']}(" + ",".join(kids) + ")"
        if n["node"] == "not":
            return f"not({rec(n['arg'])})"
        if n["node"] == "rel":
            c = n.get("const")
            return f"rel:{n['name']}({','.join(n['args'])}|{c})"
        if n["node"] == "type":
            return f"type:{n['obj']}={n['type']}"
        if n["node"] == "property":
            return f"prop:{n['obj']}={n['value']}"
        if n["node"] == "default":
            return f"default:{n['obj']}"
        if n["node"] == "choice":
            opts = sorted((f"{o.get('prior')}:{rec(o['formula']) if o.get('formula') else 'SKIP'}")
                          for o in n["options"])
            return f"choice:{n['kind']}(" + ",".join(opts) + ")"
        return json.dumps(n, sort_keys=True)
    return rec(formula_json)


def exact_match(pred: Spec, gold_json: dict) -> bool:
    return _canon(spec_to_json(pred)["formula"]) == _canon(gold_json["formula"])


def _choices_of(formula_json) -> list:
    out = []
    def rec(n):
        if isinstance(n, dict):
            if n.get("node") == "choice":
                out.append(n)
            for v in n.values():
                rec(v)
        elif isinstance(n, list):
            for x in n:
                rec(x)
    rec(formula_json)
    return out


def _nonchoice_rels(formula_json) -> set:
    """Relation leaves that are NOT inside any choice node — the 'hard' skeleton."""
    sig = set()
    def rec(n, in_choice):
        if isinstance(n, dict):
            if n.get("node") == "choice":
                for o in n.get("options", []):
                    if o.get("formula"):
                        rec(o["formula"], True)
                return
            if n.get("node") == "rel" and not in_choice:
                sig.add((n["name"], tuple(n.get("args", []))))
            for v in n.values():
                rec(v, in_choice)
        elif isinstance(n, list):
            for x in n:
                rec(x, in_choice)
    rec(formula_json, False)
    return sig


def choice_structural_match(pred: Spec, gold_json: dict,
                            require_same_options: bool = False) -> bool:
    """User-facing correctness for AMBIGUOUS specs. Matches when:
      (1) same number of CHOICE nodes, each gold CHOICE paired to a pred CHOICE with the same
          `kind` and `span`;
      (2) geometric CHOICEs (direction/offset/reference): pred's option relations overlap gold's
          on at least one reading (priors and option ordering ignored — not user-visible);
      (3) unsupported_type CHOICEs: existence of the kind+span is enough (specific suggested
          types/priors may differ — any sensible in-vocab suggestion + SKIP is acceptable);
      (4) the non-CHOICE skeleton relations match as a set.
    This reflects what a USER would judge correct, not exact prior arithmetic."""
    pf = spec_to_json(pred)["formula"]
    gc = _choices_of(gold_json["formula"])
    pc = _choices_of(pf)
    if len(gc) != len(pc):
        return False
    if _nonchoice_rels(pf) != _nonchoice_rels(gold_json["formula"]):
        return False
    used = set()
    for g in gc:
        found = False
        for i, p in enumerate(pc):
            if i in used or p.get("kind") != g.get("kind") or p.get("span") != g.get("span"):
                continue
            if g["kind"] == "unsupported_type":
                found = True; used.add(i); break
            g_rels = {(o["formula"]["name"], tuple(o["formula"]["args"]))
                      for o in g["options"] if o.get("formula") and o["formula"].get("node") == "rel"}
            p_rels = {(o["formula"]["name"], tuple(o["formula"]["args"]))
                      for o in p["options"] if o.get("formula") and o["formula"].get("node") == "rel"}
            ok = (g_rels == p_rels) if require_same_options else (len(g_rels & p_rels) >= 1)
            if ok:
                found = True; used.add(i); break
        if not found:
            return False
    return True


def is_correct(pred: Spec, gold_json: dict) -> bool:
    """Unified user-facing correctness: exact OR semantic (unambiguous) OR structural (CHOICE)."""
    gf = gold_json["formula"]
    pf = spec_to_json(pred)["formula"]
    gold_has_choice = "choice" in json.dumps(gf)
    pred_has_choice = "choice" in json.dumps(pf)
    if gold_has_choice or pred_has_choice:
        return choice_structural_match(pred, gold_json)
    return exact_match(pred, gold_json) or semantic_equiv(pred, gold_json)


def semantic_equiv(pred: Spec, gold_json: dict, n_samples: int = 200, seed: int = 0) -> bool:
    """Approximate logical equivalence of two CHOICE-free specs by agreement on random
    layouts. Sound for 'not equivalent' if a witness is found; 'equivalent' is probabilistic.
    Only valid when neither side has CHOICE (resolve first)."""
    pf = spec_to_json(pred)["formula"]
    gf = gold_json["formula"]
    if "choice" in json.dumps(pf) or "choice" in json.dumps(gf):
        return exact_match(pred, gold_json)   # fall back; equivalence undefined w/ CHOICE
    from .json_io import _node_from_json
    from .feasibility import model_check
    rng = random.Random(seed)
    ids = [o["id"] for o in gold_json["objects"]]
    existing = {o["id"]: tuple(o["box"]) for o in gold_json["objects"] if o.get("box")}
    pnode = _node_from_json(pf); gnode = _node_from_json(gf)
    for _ in range(n_samples):
        M = {}
        for oid in ids:
            if oid in existing:
                M[oid] = existing[oid]
            else:
                M[oid] = (rng.randint(0, 1000), rng.randint(0, 1000),
                          rng.randint(1, 1000), rng.randint(1, 1000))
        if model_check(pnode, M) != model_check(gnode, M):
            return False
    return True


# ---------------------------------------------------------------------------
# 1 + 2. parse accuracy / round-trip
# ---------------------------------------------------------------------------
@dataclass
class ParseReport:
    n: int
    exact: int
    semantic: int
    parse_errors: int
    def summary(self):
        return (f"parse n={self.n}  exact={self.exact}/{self.n} "
                f"({self.exact/self.n:.1%})  semantic={self.semantic}/{self.n} "
                f"({self.semantic/self.n:.1%})  errors={self.parse_errors}")


def parse_accuracy(samples: List[dict], backend) -> ParseReport:
    exact = sem = err = 0
    for s in samples:
        try:
            pred = parse(s["english"], s["objects"], backend=backend)
        except Exception:
            err += 1
            continue
        gold = {"objects": s["objects"], "formula": s["formula"]}
        if exact_match(pred, gold):
            exact += 1; sem += 1
        elif is_correct(pred, gold):    # semantic (unambiguous) or structural (CHOICE)
            sem += 1
    return ParseReport(len(samples), exact, sem, err)


# round-trip is parse_accuracy where 'english' was produced from the gold (already the case
# for synthetic data) — provided as a named entry point for clarity.
def roundtrip_consistency(samples: List[dict], backend) -> ParseReport:
    return parse_accuracy(samples, backend)


# ---------------------------------------------------------------------------
# 3. dialogue-policy evaluation
# ---------------------------------------------------------------------------
def _gold_oracle(gold_choice_answers: Dict[str, int]) -> Callable[[Question], int]:
    """Simulated user: answers each question by the gold-intended option index, keyed by
    (kind, span). Falls back to max-prior (index of highest-prior option) if unknown."""
    def oracle(q: Question) -> int:
        key = f"{q.kind}|{q.span}"
        if key in gold_choice_answers:
            return gold_choice_answers[key]
        # default: pick first (gold generators put the intended reading discoverable via prior)
        return 0
    return oracle


def _intended_answers(gold_json: dict, rng: Optional[random.Random] = None) -> Dict[str, int]:
    """Ground-truth intended reading per CHOICE.

    Crucially this is NOT always the max-prior option — otherwise ASK_NONE (which always
    takes max prior) would be trivially perfect and asking could never help. We model a
    realistic user whose intent agrees with the prior most of the time but deviates for a
    fraction of cases (where a clarifying question is exactly what's needed)."""
    rng = rng or random.Random(0)
    ans = {}
    def rec(n):
        if isinstance(n, dict) and n.get("node") == "choice":
            priors = [o.get("prior", 0) for o in n["options"]]
            mx = priors.index(max(priors))
            # 65% of the time intent == prior; else a different option is intended
            if rng.random() < 0.65 or len(priors) == 1:
                ans[f"{n['kind']}|{n.get('span','')}"] = mx
            else:
                alts = [i for i in range(len(priors)) if i != mx]
                ans[f"{n['kind']}|{n.get('span','')}"] = rng.choice(alts)
            for o in n["options"]:
                if o.get("formula"):
                    rec(o["formula"])
        elif isinstance(n, dict):
            for v in n.values():
                if isinstance(v, dict): rec(v)
                elif isinstance(v, list):
                    for x in v: rec(x)
    rec(gold_json["formula"])
    return ans


def _apply_intended(gold_json: dict, intended: Dict[str, int]) -> dict:
    """Collapse every CHOICE in gold to its INTENDED option -> the ground-truth resolved spec."""
    def rec(n):
        if isinstance(n, dict) and n.get("node") == "choice":
            idx = intended.get(f"{n['kind']}|{n.get('span','')}", 0)
            opt = n["options"][idx]
            if opt.get("formula") is None:
                return None  # SKIP
            return rec(opt["formula"])
        if isinstance(n, dict) and n.get("node") in ("and", "or"):
            kids = [rec(a) for a in n["args"]]
            kids = [k for k in kids if k is not None]
            if not kids:
                return None
            return kids[0] if len(kids) == 1 else {"node": n["node"], "args": kids}
        if isinstance(n, dict) and n.get("node") == "not":
            inner = rec(n["arg"])
            return {"node": "not", "arg": inner} if inner else None
        return n
    return {"objects": gold_json["objects"], "formula": rec(gold_json["formula"])}


@dataclass
class PolicyReport:
    n: int
    policy_questions: float
    askall_questions: float
    policy_correct: int
    asknone_correct: int
    askall_correct: int
    def summary(self):
        return (f"policy n={self.n}  "
                f"avg Q: policy={self.policy_questions:.2f}  ASK_ALL={self.askall_questions:.2f}  ASK_NONE=0.00\n"
                f"     correct vs true intent: "
                f"policy={self.policy_correct}/{self.n} ({_pct(self.policy_correct,self.n)})  "
                f"ASK_ALL={self.askall_correct}/{self.n} ({_pct(self.askall_correct,self.n)})  "
                f"ASK_NONE={self.asknone_correct}/{self.n} ({_pct(self.asknone_correct,self.n)})")


def _pct(a, b):
    return f"{(a/max(1,b)):.0%}"


def policy_eval(samples: List[dict], budget: int = 3, seed: int = 0) -> PolicyReport:
    rng = random.Random(seed)
    ambig = [s for s in samples if s.get("ambiguous")]
    pol_q = askall_q = 0
    pol_ok = none_ok = all_ok = 0
    for s in ambig:
        gold_json = {"objects": s["objects"], "formula": s["formula"]}
        intended = _intended_answers(gold_json, rng)
        truth = _apply_intended(gold_json, intended)           # ground-truth resolved spec
        oracle = _gold_oracle(intended)                        # user answers per true intent

        # --- our policy ---
        spec = spec_from_json(gold_json)
        _, log = resolve(spec, oracle=oracle, budget=budget)
        pol_q += len(log.asked)
        if exact_match(spec, truth):
            pol_ok += 1

        # --- ASK_ALL (confirm every choice -> always matches intent, max questions) ---
        spec_all = spec_from_json(gold_json)
        _, n_all = resolve_ask_all(spec_all, oracle=oracle)
        askall_q += n_all
        if exact_match(spec_all, truth):
            all_ok += 1

        # --- ASK_NONE (max prior, never ask -> wrong whenever intent != prior) ---
        spec_none = spec_from_json(gold_json)
        resolve_ask_none(spec_none)
        if exact_match(spec_none, truth):
            none_ok += 1

    n = max(1, len(ambig))
    return PolicyReport(len(ambig), pol_q / n, askall_q / n, pol_ok, none_ok, all_ok)
