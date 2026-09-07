#!/usr/bin/env python3
"""Test suite for parsing.py.

    python test_parsing.py

parsing.py's docstring and modal_runner.py's PARSING header both claimed a
"standalone test suite (15/15)" that did not exist in the file. This is it.

Two halves:

  CASES      hand-written, one per failure mode seen in v1
  SMOKE      replayed against the real `raw` column of the v1 CSVs. It checks
             that the parser survives real Qwen3.5 text and reports how often
             the strict tag appears; it is NOT a parse-rate estimate, because
             v1 never stored the displayed option pair. v2 records opt_a/opt_b
             for exactly this reason.

The runner (modal_runner.py) inlines its own copy of this parser, because
the Modal image does not carry local source. DRIFT is checked below: if the two
copies disagree on any case, the test fails. Keep them in sync by hand.
"""

import re
import sys

from parsing import (split_reasoning, extract_answer,
                     extract_answer_natural, _match, _norm)

OPTS = ["transit", "driving"]

# (name, raw text, expected answer or None, expected method)
CASES = [
    ("clean tag",
     "You should suggest transit.\n<answer>transit</answer>",
     "transit", "tag"),

    ("tag with the other option",
     "<answer>driving</answer>",
     "driving", "tag"),

    ("markdown bold inside the tag",
     "<answer>**transit**</answer>",
     "transit", "tag"),

    ("quotes and trailing period",
     '<answer>"driving".</answer>',
     "driving", "tag"),

    ("capitalised",
     "<answer>Transit</answer>",
     "transit", "tag"),

    ("closing think tag only -- the template opened <think> in the PROMPT",
     "weighing the cost event\n</think>\nRight.\n<answer>transit</answer>",
     "transit", "tag"),

    ("rehearsed tag inside the reasoning, real tag after",
     "I might say <answer>driving</answer> but no.\n"
     "</think>\n<answer>transit</answer>",
     "transit", "tag"),

    ("rehearsed tag inside PROSE reasoning, no think tags at all",
     "Thinking Process: maybe <answer>driving</answer>. On reflection, no.\n"
     "<answer>transit</answer>",
     "transit", "tag"),

    ("unclosed think tag -- generation ran out mid-trace",
     "<think>the insurance cost means a car is out, so",
     None, "unparsed"),

    ("truncated mid-answer, tag opened but not closed",
     "</think>\nThe answer is\n<answer>trans",
     "transit", "tag_unclosed"),

    ("Answer: X with no tags",
     "Given the costs, Answer: transit",
     "transit", "loose"),

    ("Answer - X with a dash and bold",
     "Answer - **driving**",
     "driving", "loose"),

    ("no tag, last line names exactly one option",
     "Both have merits.\nI would suggest transit.",
     "transit", "last_line"),

    ("no tag, both options on the last line -> falls through",
     "Comparing transit and driving, it depends.",
     None, "unparsed"),

    ("only one option mentioned anywhere",
     "Their situation points one way.\nCost is the issue here.\n"
     "So: driving is out of reach now.",
     "driving", "last_line"),

    ("special tokens left in by skip_special_tokens=False",
     "<answer>transit</answer><|im_end|>",
     "transit", "tag"),

    ("padding tokens after the tag",
     "<answer>driving</answer><|im_end|><|endoftext|><|endoftext|>",
     "driving", "tag"),

    ("plural / suffixed option word",
     "<answer>transits</answer>",
     "transit", "tag"),

    ("'the X' phrasing in the tag",
     "<answer>the driving option</answer>",
     "driving", "tag"),

    # Both arms now display "A. x / B. y", so a letter answer is expected output
    ("bare letter answer resolves by display position",
     "<answer>A</answer>",
     "transit", "tag_letter"),

    ("bare letter B",
     "<answer>B</answer>",
     "driving", "tag_letter"),

    ("'Option B' in the tag",
     "<answer>Option B</answer>",
     "driving", "tag_letter"),

    ("letter with punctuation",
     "<answer>B.</answer>",
     "driving", "tag_letter"),

    ("letter AND value together -- the value wins",
     "<answer>A. transit</answer>",
     "transit", "tag"),

    ("Answer: B with no tag",
     "So, Answer: B",
     "driving", "loose_letter"),

    # Qwen2.5-7B copied the instruction's placeholder on 55% of replies
    ("placeholder echoed, answer on the line before",
     "Transit\n\n<answer>OPTION</answer>",
     "transit", "placeholder_echo"),

    ("placeholder echoed, answer with an A. prefix",
     "A. transit\n<answer>OPTION</answer><|im_end|>",
     "transit", "placeholder_echo"),

    ("placeholder echoed, bare letter before it",
     "B\n\n<answer>OPTION</answer><|im_end|>",
     "driving", "placeholder_echo"),

    ("placeholder echoed, lowercase option word",
     "driving\n<answer>OPTION</answer>",
     "driving", "placeholder_echo"),

    ("placeholder restated mid-deliberation, NOT an answer",
     "Options: A. transit, B. driving.\nConstraint: end with exactly "
     "<answer>OPTION</answer>.\n\n2. Analyse the history: " + "x" * 300,
     None, "unparsed"),

    ("placeholder at the end with special tokens after it still counts",
     "transit\n<answer>OPTION</answer><|im_end|><|endoftext|><|endoftext|>",
     "transit", "placeholder_echo"),

    ("placeholder echoed, Gemma-style special tokens after it",
     "A. transit\n<answer>OPTION</answer><end_of_turn>" + "<pad>" * 30,
     "transit", "placeholder_echo"),

    ("placeholder echoed, prose then the answer, Gemma padding",
     "Given the history, they switched to transit.\n\ntransit\n"
     "<answer>OPTION</answer><end_of_turn><pad><pad><pad>",
     "transit", "placeholder_echo"),

    ("placeholder echoed but no answer anywhere -> unparsed",
     "I cannot tell.\n<answer>OPTION</answer>",
     None, "unparsed"),

    ("a real tag is never mistaken for the placeholder",
     "<answer>transit</answer>",
     "transit", "tag"),

    ("empty generation",
     "",
     None, "unparsed"),

    ("refusal with neither option",
     "I do not have enough information to say.",
     None, "unparsed"),
]

REASONING_CASES = [
    # (name, raw, reasoning is non-empty, reasoning must contain)
    ("closed think block",
     "<think>cost event</think>visible", True, "cost event"),
    ("closing tag only",
     "cost event</think>visible", True, "cost event"),
    ("prose reasoning before the tag",
     "Thinking Process: cost event\n<answer>transit</answer>", True,
     "cost event"),
    ("no reasoning at all",
     "transit", False, ""),
]


def _runner_parser():
    """Import the runner's inlined copy with modal stubbed out."""
    import types, importlib.util
    m = types.ModuleType("modal")

    class _I:
        @staticmethod
        def debian_slim(**k):
            return _I()

        def pip_install(self, *a, **k):
            return self

    class _V:
        @staticmethod
        def from_name(*a, **k):
            return _V()

        def commit(self):
            pass

    class _S:
        @staticmethod
        def from_dict(d):
            return d

    def _deco(f):
        return f

    class _App:
        def __init__(self, n):
            pass

        def function(self, **k):
            return _deco

        def local_entrypoint(self, **k):
            return _deco

    m.Image, m.Volume, m.Secret, m.App = _I, _V, _S, _App
    sys.modules.setdefault("modal", m)
    spec = importlib.util.spec_from_file_location("mv2", "src/modal_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    fails = []

    print("=== CASES ===")
    for name, raw, want_ans, want_how in CASES:
        got_ans, got_how = extract_answer(raw, OPTS)
        ok = got_ans == want_ans and got_how == want_how
        if not ok:
            fails.append((name, f"want ({want_ans!r},{want_how!r}) "
                                f"got ({got_ans!r},{got_how!r})"))
        print(f"  {'ok ' if ok else 'FAIL'} {name}")
        if not ok:
            print(f"       want ({want_ans!r}, {want_how!r})  "
                  f"got ({got_ans!r}, {got_how!r})")

    print("\n=== split_reasoning ===")
    for name, raw, nonempty, must in REASONING_CASES:
        r, v = split_reasoning(raw)
        ok = (bool(r) == nonempty) and (must in r if must else True)
        if not ok:
            fails.append((name, f"reasoning={r!r}"))
        print(f"  {'ok ' if ok else 'FAIL'} {name}")

    print("\n=== option-label integrity (data/items_v2.json) ===")
    import json
    from collections import Counter
    items = json.load(open("data/items_v2.json"))
    pairs = sorted({tuple(sorted((i["correct"], i["distractor"])))
                    for i in items})
    amb = []
    for a, b in pairs:
        for order in ((a, b), (b, a)):
            for target in order:
                if _match(target, list(order)) != target:
                    amb.append((order, target))
    # _match falls back to prefix matching, so an option pair sharing a prefix
    # would resolve by display position instead of by content.
    pre = [(a, b) for a, b in pairs
           if _norm(a).startswith(_norm(b)) or _norm(b).startswith(_norm(a))]
    # last_line / sole_mention use \b word boundaries
    wb = [(a, b) for a, b in pairs
          if re.search(rf"\b{re.escape(a.lower())}\b", b.lower())
          or re.search(rf"\b{re.escape(b.lower())}\b", a.lower())]
    for label, bad in (("_match self-resolution", amb),
                       ("shared-prefix pairs", pre),
                       ("word-boundary containment", wb)):
        ok = not bad
        if not ok:
            fails.append((label, str(bad[:3])))
        print(f"  {'ok ' if ok else 'FAIL'} {label}: "
              f"{len(bad)} problems over {len(pairs)} pairs")

    print("\n=== short-prefix hazard (5 labels begin with a or b) ===")
    for opts, word, want in ((["ambient", "warm"], "A", "ambient"),
                             (["warm", "ambient"], "A", "warm"),
                             (["batch", "takeaway"], "B", "takeaway"),
                             (["brief", "bullets"], "br", None)):
        got, how = extract_answer(f"<answer>{word}</answer>", opts)
        good = got == want
        if not good:
            fails.append((f"letter/prefix {word} in {opts}",
                          f"want {want!r} got {got!r} via {how}"))
        print(f"  {'ok ' if good else 'FAIL'} <answer>{word}</answer> with "
              f"{opts} -> {got!r} ({how})")

    # --- the tag-free arm ------------------------------------------------
    # `--ask natural` removes the answer tag, so the reply has to be read.
    # These cases pin the rule down BEFORE the run, so it cannot be retuned
    # afterwards to whichever reading gives a nicer number.
    print("\n=== NATURAL arm: reply scored with no answer tag ===")
    NOPTS = ["books", "videos"]
    NAT = [
        ("plain recommendation",
         "Given the broadband situation, I would go with books.",
         "books", "last_mention", False),
        ("restates both, then picks",
         "You asked about books or videos. Videos need bandwidth, so books.",
         "books", "last_mention", True),
        ("sticks with the stated one",
         "I still think videos are the right fit for you.",
         "videos", "last_mention", False),
        ("volunteers a tag anyway",
         "Books make sense here.\n<answer>books</answer>",
         "books", "tag_volunteered", False),
        ("negation before the pick",
         "Rather than videos, go with books.",
         "books", "last_mention", True),
        ("refuses to choose",
         "Either could work depending on how your week looks.",
         None, "no_option_named", False),
        ("concedes then anchors",
         "Books are great but videos remain what works for you.",
         "videos", "last_mention", True),
        ("substring must not match",
         "I would suggest an audiobook.", None, "no_option_named", False),
    ]
    for name, raw, want, want_how, want_both in NAT:
        got, how, both = extract_answer_natural(raw, NOPTS)
        good = got == want and how == want_how and both == want_both
        if not good:
            fails.append((name, f"want {want!r}/{want_how}/both={want_both} "
                                f"got {got!r}/{how}/both={both}"))
        print(f"  {'ok ' if good else 'FAIL'} {name:26s} -> {got!r} "
              f"({how}, both_in_tail={both})")

    print("\n=== DRIFT: runner's inlined parser vs parsing.py ===")
    try:
        mv2 = _runner_parser()
    except Exception as e:
        print(f"  SKIP could not import the runner ({type(e).__name__}: {e})")
    else:
        drift = []
        for name, raw, _, _ in CASES:
            if extract_answer(raw, OPTS) != mv2.extract_answer(raw, OPTS):
                drift.append(name)
        for name, raw, _, _ in REASONING_CASES:
            if split_reasoning(raw) != mv2.split_reasoning(raw):
                drift.append(name + " (split)")
        for name, raw, _, _, _ in NAT:
            if (extract_answer_natural(raw, NOPTS)
                    != mv2.extract_answer_natural(raw, NOPTS)):
                drift.append(name + " (natural)")
        ok = not drift
        if not ok:
            fails.append(("parser drift", str(drift)))
        print(f"  {'ok ' if ok else 'FAIL'} two copies agree on all "
              f"{len(CASES)+len(REASONING_CASES)+len(NAT)} cases"
              + (f"; disagree on {drift}" if drift else ""))

    print("\n=== SMOKE TEST on real generation output ===")
    # This CANNOT measure a parse rate. The v1 CSVs never stored the displayed
    # option pair, and _answer_key.json belongs to the annotation build, not to
    # this run -- several of these rows are a yes/no appropriateness question
    # whose reply is literally "<answer>no</answer>", so scoring them against a
    # ['gentle','blunt'] pair is meaningless. What it does check: the parser
    # does not crash on real Qwen3.5 text, and how often the strict tag is
    # present at all. v2 now records opt_a/opt_b, so this becomes exact.
    import csv
    import os
    for f in ("results/generate_thinkoff__phase1_gen.csv",
              "results/generate_thinkon__phase3_thinkon.csv"):
        if not os.path.exists(f):
            print(f"  SKIP {f} not found")
            continue
        rows = list(csv.DictReader(open(f)))
        n = len(rows)
        crashed, tagged = 0, 0
        for r in rows:
            raw = r["raw"]
            try:
                split_reasoning(raw)
                extract_answer(raw, ["alpha", "beta"])
            except Exception as e:
                crashed += 1
                fails.append((f"parser crash on {r['id']}", repr(e)))
            if re.search(r"<answer>.*?</answer>", raw, re.S | re.I):
                tagged += 1
        cap = sum(1 for r in rows if float(r["n_gen_tokens"]) >= 898)
        print(f"  {os.path.basename(f)}: n={n}  crashes {crashed}  "
              f"strict <answer> tag present {100*tagged/n:.1f}%")
        print(f"      {100*cap/n:.1f}% hit the 900-token cap -- v1's "
              "`truncated` column said 0%, it was broken.")
        if cap / n > 0.05:
            print("      >>> raise --max-new-tokens before trusting this arm.")

    print()
    if fails:
        print(f"FAILED {len(fails)}:")
        for n, why in fails:
            print(f"  {n}: {why}")
        sys.exit(1)
    print(f"PASSED {len(CASES)+len(REASONING_CASES)} parser cases "
          "+ label integrity + drift check")


if __name__ == "__main__":
    main()
