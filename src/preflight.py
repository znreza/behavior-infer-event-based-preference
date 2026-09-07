#!/usr/bin/env python3
"""Everything that can be checked before spending a GPU-minute.

    python preflight.py

Run this after any edit to anchors_full.json, generate_v2.py or
modal_runner.py, and before starting run_all.sh. Exit code 1 means do not
launch. Warnings (marked WARN) are judgement calls, not blockers.

Grouped by what each check protects:

  ENV        the Modal local_entrypoint runs on THIS machine, not in the image
  DATA       data/items_v2.json is exactly what anchors_full.json implies
  DESIGN     the condition contrast is the one the paper claims
  LEAKAGE    the answer cannot be had without reading the event
  RENDERING  the runner shows the model what the generator says it shows
  RESOURCES  the run fits in the GPU and the timeout
  PLAN       the emitted commands point at files that exist and do not collide
  RETENTION  the CSV a run produces can still answer the run's own question
"""

import csv
import json
import os
import random
import re
import sys
import types
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
# Every data path below is relative to the repo root, so pin the working
# directory there rather than to src/. This makes the script runnable from
# anywhere.
os.chdir(os.path.dirname(HERE))

ANCHORS = "data/anchors/anchors_full.json"
ITEMS = "data/items_v2.json"
RUNNER = "src/modal_runner.py"

FAIL, WARN = [], []


def ok(msg):
    print(f"  ok   {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"  FAIL {msg}")


def warn(msg):
    WARN.append(msg)
    print(f"  WARN {msg}")


def head(t):
    print(f"\n=== {t} ===")


def _stub_modal():
    if "modal" in sys.modules:
        return
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
    sys.modules["modal"] = m


def load_runner():
    _stub_modal()
    import importlib.util
    spec = importlib.util.spec_from_file_location("mv2", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeTok:
    """Stands in for a real tokenizer: same structure, no download."""

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True, **kw):
        return ("<|im_start|>user\n" + msgs[0]["content"]
                + "<|im_end|>\n<|im_start|>assistant\n")


# ===========================================================================
def check_env():
    head("ENV -- main() is a local_entrypoint, it runs here")
    print(f"  python {sys.version.split()[0]}")
    for mod, why in (("modal", "required to launch anything"),
                     ("pandas", "main() imports it locally to write the CSV")):
        try:
            __import__(mod)
            ok(f"{mod} importable")
        except ImportError:
            fail(f"{mod} is missing locally ({why}) -> "
                 f"pip install 'modal>=0.64' pandas")
    sys.argv = ["preflight"]
    import run_plan as _P
    gated = sorted({m for p in _P.PLAN for m in p.get("models", [])
                    if any(g in m.lower() for g in ("gemma", "llama",
                                                    "mistral"))})
    if not os.environ.get("HF_TOKEN"):
        if gated:
            fail(f"HF_TOKEN is unset and the plan contains GATED repos "
                 f"{gated}. Every load will 401 inside the container. Accept "
                 "the licence on the model page, then export HF_TOKEN before "
                 "launching -- the runner passes it through as a Modal secret.")
        else:
            warn("HF_TOKEN unset. Fine for public Qwen weights; a gated repo "
                 "would fail at load with a 401 inside the container.")
    elif gated:
        ok(f"HF_TOKEN set; gated repos in the plan: {gated}")


def check_data():
    head("DATA -- data/items_v2.json is reproducible from the anchors")
    sys.argv = ["preflight"]
    import generate_v2 as G
    attrs = G.load_anchors(ANCHORS)
    items = json.load(open(ITEMS))
    frames = sorted({i.get("setup_frame", "requirement") for i in items},
                    key=lambda f: list(G.SETUP_FRAMES).index(f))
    regen = G.build_items(attrs, sorted({i["n_turns"] for i in items}),
                          sorted({i["event_pos"] for i in items}),
                          n_variants=1, seed=0, frames=tuple(frames))
    if len(regen) != len(items):
        fail(f"{ITEMS} has {len(items)} items, the anchors imply {len(regen)} "
             f"for frames={frames}. Regenerate: python src/generate_v2.py "
             f"--anchors {ANCHORS} --variants 1 --frames "
             f"{','.join(frames)} --out {ITEMS}")
        return items, attrs
    diff = [(a["id"], k) for a, b in zip(regen, items)
            for k in a if a[k] != b[k]]
    if diff:
        fields = Counter(k for _, k in diff)
        fail(f"{ITEMS} is stale in {len(diff)} fields {dict(fields)}. "
             "Regenerate it.")
    else:
        ok(f"{len(items)} items byte-identical to a fresh build "
           f"({len(attrs)} attributes, "
           f"{sum(len(a['events']) for a in attrs)} anchor events, "
           f"frames={frames})")

    if len({i["id"] for i in items}) != len(items):
        fail("duplicate item ids")
    else:
        ok("item ids unique")
    return items, attrs


def check_design(items):
    head("DESIGN -- the cells are balanced and the key is what it claims")
    ct = Counter((i["condition"], i["n_turns"], i["event_pos"]) for i in items)
    sizes = set(ct.values())
    if len(sizes) == 1:
        ok(f"every (condition, turns, event_pos) cell has {sizes.pop()} items "
           f"({len({c for c, *_ in ct})} conditions x "
           f"{len({t for _, t, _ in ct})} turn counts x "
           f"{len({p for *_, p in ct})} positions = {len(ct)} cells)")
    else:
        fail(f"unbalanced cells: sizes {sorted(sizes)}")

    # T0 must be the only condition where nothing happened
    for cond, grp in _by(items, "condition").items():
        nulls = sum(1 for i in grp if i["event_text"] is None)
        if cond.startswith("T0") and nulls != len(grp):
            fail(f"{cond}: {len(grp)-nulls} items DO carry an event")
        if not cond.startswith("T0") and nulls:
            fail(f"{cond}: {nulls} items carry no event")
    ok("T0 carries no event; T1-T4 all do")

    # the correct answer is always the post-event state
    bad = [i["id"] for i in items if i["correct"] != i["new"]]
    if bad:
        fail(f"{len(bad)} items where correct != new")
    else:
        ok("correct == new for every item")

    # T1/T2/T4 must offer the old state as the distractor; T3 must not
    for cond in ("T1_explicit", "T2_implicit", "T4_conflict"):
        grp = _by(items, "condition").get(cond, [])
        n = sum(1 for i in grp if i["distractor"] == i["old"])
        if grp and n != len(grp):
            fail(f"{cond}: distractor is the old state on only {n}/{len(grp)}")
    t3 = _by(items, "condition").get("T3_counterfact", [])
    n = sum(1 for i in t3 if i["old"] in (i["correct"], i["distractor"]))
    if n:
        fail(f"T3: the old state is a displayed option on {n} items")
    elif t3:
        ok("T1/T2/T4 offer the old state as the distractor; T3 never does "
           "(so T3 cannot show stale-preference errors -- report it separately)")

    srcs = Counter(i["source"] for i in items)
    origins = Counter(i.get("origin", "?") for i in items)
    if len(srcs) == 1:
        ok(f"`source` is uniform ({list(srcs)[0]}): every anchor is "
           f"author-validated, so no result is split on it")
    else:
        warn(f"`source` is not uniform ({dict(srcs)}); the plan no longer "
             "splits on it")
    if len(origins) > 1:
        ok(f"`origin` preserves who drafted first {dict(origins)} -- "
           "recoverable, but a diagnostic, not a robustness split")
    else:
        warn("`origin` is missing from the items; the author/LLM-drafted "
             "distinction is no longer recoverable")

    # T3 metadata must describe the event actually shown
    if t3 and "shown_event_id" in t3[0]:
        wrong = sum(1 for i in t3 if i["shown_event_id"] == i["event_id"])
        ok(f"T3 records shown_event_id; {len(t3)-wrong}/{len(t3)} differ from "
           "the anchor id, as they should")
    elif t3:
        fail("T3 items have no shown_event_id: source and event_type still "
             "describe the anchor event, not the one displayed. Regenerate "
             "with the patched generate_v2.py.")


def check_leakage(items):
    head("LEAKAGE -- the answer must require reading the event")
    sys.argv = ["preflight"]
    import generate_v2 as G
    attrs = G.load_anchors(ANCHORS)
    leaks = G.check_leaks(attrs, verbose=False)
    if leaks:
        fail(f"{len(leaks)} event realizations name a state label: {leaks[:3]}")
    else:
        ok("no state label appears in any event realization")

    # the same option pair must be correct in both directions somewhere in the
    # set, or "which two options appear" gives the answer away
    d = defaultdict(Counter)
    for i in items:
        d[tuple(sorted((i["correct"], i["distractor"])))][i["correct"]] += 1
    one = [k for k, v in d.items() if len(v) == 1]
    if one:
        fail(f"{len(one)}/{len(d)} option pairs are only ever correct one way: "
             f"{one[:3]} -- a model can score above chance without reading")
    else:
        ok(f"all {len(d)} option pairs are correct in both directions")

    # the event text must never contain either displayed option
    hits = [i["id"] for i in items if i["event_text"] and any(
        re.search(rf"\b{re.escape(o.lower())}\b", i["event_text"].lower())
        for o in (i["correct"], i["distractor"]))]
    t1 = [h for h in hits if "T1_explicit" in h]
    other = [h for h in hits if "T1_explicit" not in h]
    if other:
        fail(f"{len(other)} non-T1 items name an option in the event text: "
             f"{other[:3]}")
    else:
        ok(f"only T1 states an option outright ({len(t1)} items, by design)")

    # the setup names the old state; for T1/T2/T4 that is the distractor, so a
    # model that copies the setup scores 0 on those and 100 on T0
    ok("T0 and T2 share an option pair with opposite answers -- copying the "
       "setup gives 100% on T0 and 0% on T2")


def check_rendering(items):
    head("RENDERING -- the runner shows what the generator says it shows")
    sys.argv = ["preflight"]
    import generate_v2 as G
    mv2 = load_runner()

    bad = [i["id"] for i in items
           if G.build_conversation(i, i["n_turns"], i["event_pos"],
                                   random.Random(i["seed"]))[0]
           != mv2.build_conversation(i)[0]]
    if bad:
        fail(f"{len(bad)} conversations differ between generate_v2 and the "
             f"runner: {bad[:3]}")
    else:
        ok(f"all {len(items)} conversation bodies identical in both modules")

    tok = FakeTok()
    it = next(i for i in items if i["condition"] == "T2_implicit"
              and i["n_turns"] == 4)
    ref, _, _, _ = G.render(it, 0)
    probe, _ = mv2.probe_prompt(it, 0, tok, "chat_prefill")
    gen, _ = mv2.gen_prompt(it, 0, tok, False)
    if "\nA. " not in probe:
        fail("probe prompt has no A./B. option lines")
    else:
        ok("probe prompt enumerates the options as A./B.")
    # the two arms must differ only in the response instruction
    bad = []
    for cond in sorted({i["condition"] for i in items}):
        c = next(i for i in items if i["condition"] == cond
                 and i["n_turns"] == 4 and i["event_pos"] == 0.6)
        for o in (0, 1):
            ref, _, _, _ = G.render(c, o)
            g, _ = mv2.gen_prompt(c, o, tok, False)
            inner = g.split("<|im_start|>user\n", 1)[1].rsplit("<|im_end|>", 1)[0]
            if inner != ref:
                bad.append(f"{cond}/order{o}")
    if bad:
        fail(f"gen_prompt differs from generate_v2.render() on {bad}")
    else:
        ok("gen_prompt is byte-identical to generate_v2.render() on every "
           "condition and both option orders")

    # The ask frames are a second copy of the stimulus logic living in two
    # modules, which is exactly how the probe and generation arms drifted
    # apart the first time. Check every frame, not just the default.
    if set(G.ASK_FRAMES) != set(mv2.ASK_FRAMES):
        fail(f"ASK_FRAMES differ: {sorted(G.ASK_FRAMES)} vs "
             f"{sorted(mv2.ASK_FRAMES)}")
    else:
        ok(f"both modules define the same ask frames "
           f"({', '.join(sorted(G.ASK_FRAMES))})")
    bad = []
    for frame in sorted(G.ASK_FRAMES):
        for cond in sorted({i["condition"] for i in items}):
            c = next(i for i in items if i["condition"] == cond
                     and i["n_turns"] == 4 and i["event_pos"] == 0.6)
            for o in (0, 1):
                ref = G.render(c, o, frame)[0]
                run = mv2.question_body(c, o, frame)[0]
                if G.ASK_FRAMES[frame]["tag"]:
                    run += mv2.ANSWER_INSTRUCTION.format(
                        a=mv2.question_body(c, o, frame)[1][0],
                        b=mv2.question_body(c, o, frame)[1][1])
                if ref != run:
                    bad.append(f"{frame}/{cond}/order{o}")
    if bad:
        fail(f"the two modules render different stimuli on {bad[:4]}")
    else:
        ok("every ask frame renders identically in both modules")

    # What each frame is supposed to remove, asserted rather than assumed.
    q0 = G.render(it, 0, "question")[0]
    fp = G.render(it, 0, "firstperson")[0]
    nat = G.render(it, 0, "natural")[0]
    checks = [("question", "asks you for a suggestion" in q0),
              ("firstperson drops the third-person question line",
               "asks you for a suggestion" not in fp),
              ("firstperson keeps the options", "\nA. " in fp),
              ("firstperson keeps the answer tag", "<answer>OPTION" in fp),
              ("natural is a turn list", isinstance(nat, list)),
              ("natural names both options in the last turn",
               isinstance(nat, list)
               and all(o in nat[-1]["content"] for o in
                       G.render(it, 0, "natural")[2])),
              ("natural has no answer tag",
               isinstance(nat, list)
               and not any("<answer>" in m["content"] for m in nat)),
              ("natural has no excerpt header",
               isinstance(nat, list)
               and not any("excerpts from your past" in m["content"]
                           for m in nat))]
    broken = [n for n, good in checks if not good]
    if broken:
        fail(f"ask frames do not do what they claim: {broken}")
    else:
        ok("each ask frame removes exactly the cues it claims to")
    pb, popts = mv2.probe_prompt(it, 0, tok, "chat_prefill")
    gb, _ = mv2.gen_prompt(it, 0, tok, False)
    shared = mv2.question_body(it, 0)[0]
    if shared in pb and shared in gb:
        ok("probe and generation share one stimulus; they differ only in the "
           "response instruction")
    else:
        fail("probe and generation no longer share question_body()")
    if probe.count("The answer is option:") > 1:
        warn("'The answer is option:' appears twice in the chat_prefill probe "
             "(once inside the user turn, once as the assistant prefill). "
             "Harmless, but say so if a reviewer asks.")

    # the event must actually be inside the rendered conversation
    miss = []
    for i in items[:800]:
        if i["event_text"] is None:
            continue
        body, _ = mv2.build_conversation(i)
        if i["event_text"] not in body:
            miss.append(i["id"])
    if miss:
        fail(f"{len(miss)} items whose event text is not in the conversation: "
             f"{miss[:3]}")
    else:
        ok("event text present in the rendered conversation (800 sampled)")

    # filler reuse at long turn counts
    for t in sorted({i["n_turns"] for i in items}):
        i = next(x for x in items if x["n_turns"] == t
                 and x["condition"] == "T2_implicit")
        body, _ = mv2.build_conversation(i)
        users = [l.split(": ", 1)[1] for l in body.split("\n")
                 if l.startswith(i["name"] + ": ")]
        dup = len(users) - len(set(users))
        if dup:
            warn(f"n_turns={t}: {dup} of {len(users)} user turns are verbatim "
                 f"repeats (the filler pool holds {len(mv2.FILLER)}). Add more "
                 "fillers or state it as a limitation of the turn ablation.")
    ok(f"no conversation repeats a turn at any length "
       f"(filler pool: {len(mv2.FILLER)}, longest run: "
       f"{max(i['n_turns'] for i in items)} turns)")
    if G.FILLER != mv2.FILLER:
        fail("FILLER differs between generate_v2 and the runner -- the two "
             "modules would render different conversations")
    labels = {s_["label"] for a in G.load_anchors(ANCHORS) for s_ in a["states"]}
    hits = [(t, l) for t in mv2.FILLER for l in labels
            if re.search(rf"\b{re.escape(l.lower())}\b", t.lower())]
    if hits:
        fail(f"{len(hits)} filler turns contain a state label: {hits[:3]}")
    else:
        ok(f"no filler turn contains any of the {len(labels)} state labels")

    # length and distance-to-question are collinear at a single event_pos
    pos = sorted({i["event_pos"] for i in items})
    if len(pos) == 1:
        warn(f"event_pos is fixed at {pos[0]}, so distance-to-question moves "
             "with length and neither can be attributed alone. Regenerate with "
             "--positions 0.3,0.6,0.9.")
    else:
        rows = []
        for t in sorted({i["n_turns"] for i in items}):
            idx = [max(2, min(t - 1, round(p * t))) for p in pos]
            rows.append(f"    n_turns={t:3d}: event at turn {idx} "
                        f"-> {[t - x for x in idx]} turns before the question")
        ok(f"event_pos varies over {pos}, so length and distance are separable")
        for r in rows:
            print(r)
        if len(set(idx for t in [4] for idx in
                   [tuple(max(2, min(t - 1, round(p * t))) for p in pos)])) and \
                len({max(2, min(3, round(p * 4))) for p in pos}) < len(pos):
            warn("at n_turns=4 the positions collapse (round(0.3*4)=1 is "
                 "clamped to 2, same as 0.6), so the position ablation needs "
                 "a longer cell -- the plan uses n_turns=20.")


def check_resources(items):
    head("RESOURCES -- does the run fit the GPU and the timeout")
    mv2 = load_runner()
    tok = FakeTok()
    V, CHARS_PER_TOK, W_9B = 151936, 3.9, 18.5
    print(f"  {'turns':>6s} {'~tok max':>9s} {'logits GB @bs16':>16s} "
          f"{'+9B weights':>12s}")
    worst = 0
    for t in sorted({i["n_turns"] for i in items}):
        mx = max(len(mv2.probe_prompt(i, 0, tok, "chat_prefill")[0])
                 for i in items if i["n_turns"] == t)
        ntok = mx / CHARS_PER_TOK
        gb = 16 * ntok * V * 2 / 1e9
        worst = max(worst, gb)
        print(f"  {t:6d} {ntok:9.0f} {gb:16.1f} {gb + W_9B:12.1f}")
    if worst + W_9B > 36:
        if "_last_logits" in open(RUNNER).read():
            ok(f"a full logits tensor would reach {worst:.1f} GB at bs16; "
               "the runner asks for last-position logits only")
        else:
            fail(f"the full logits tensor reaches {worst:.1f} GB at bs16, plus "
                 f"{W_9B} GB of 9B weights, on a 40 GB A100. Pass "
                 "logits_to_keep=1 or lower --batch-size.")
    if "jobs.sort(key=lambda j: len(j[2]))" in open(RUNNER).read():
        ok("jobs are length-sorted before batching (turn counts interleave in "
           f"{ITEMS}, so unsorted batches pay the 50-turn cost every time)")
    else:
        warn("jobs are not length-sorted; every batch in the turn ablation "
             "will mix 200-token and 1700-token prompts")

    mv2 = load_runner()
    sys.argv = ["preflight"]
    import run_plan as P
    BILLIONS = {"1b": 1, "4b": 4, "7b": 7, "8b": 8, "9b": 9, "12b": 12,
                "14b": 14, "27b": 27, "0.8b": 0.8, "2b": 2}
    CARD_GB = {"A100-40GB": 40, "A100-80GB": 80, "H100": 80, "A10G": 24,
               "L4": 24, "T4": 16}
    print("       weights vs card, per model in the plan:")
    for p_ in P.PLAN:
        card = p_.get("gpu") or mv2.GPU
        cap = CARD_GB.get(card, 40)
        for m in p_.get("models", []):
            key = next((k for k in sorted(BILLIONS, key=len, reverse=True)
                        if k in m.lower()), None)
            if not key:
                continue
            # 2 bytes per parameter in fp16/bf16, plus room for KV and activations
            need = BILLIONS[key] * 2
            if need > cap * 0.75:
                fail(f"{m} needs ~{need:.0f} GB of weights but {p_['name']!r} "
                     f"targets {card} ({cap} GB). Pass --gpu A100-80GB or H100 "
                     "for that row.")
    # A --gpu that is accepted and ignored sends a 54GB model to a 40GB card
    # and fails 6 seconds into a 54GB download. Assert every remote call in
    # main() actually routes through the override.
    # read from the file: under the real modal package main() is a
    # LocalEntrypoint object, not a function, so inspect.getsource fails
    rsrc = open(RUNNER).read()
    msrc = rsrc[rsrc.index("def main("):]
    starmaps = re.findall(r"(\w+)\.starmap\(", msrc)
    routed = re.findall(r"(\w+)\s*=\s*_with_gpu\(", msrc)
    unrouted = [x for x in starmaps if x not in routed and x != "fn"]
    if "_with_gpu" not in msrc:
        fail("main() has no --gpu override path at all")
    elif unrouted:
        fail(f"these remote calls bypass the --gpu override: {unrouted}. "
             "A gpu flag would be accepted and silently ignored.")
    else:
        ok(f"all {len(starmaps)} remote call sites route through the --gpu "
           "override")
    lsrc = rsrc[rsrc.index("def _load_model("):rsrc.index("def _letter_ids(")]
    if "empty_cache" not in lsrc:
        warn("the model loader does not free GPU memory between fallback "
             "attempts; a failed first attempt can OOM the correct loader")
    else:
        ok("the model loader frees GPU memory between fallback attempts")

    seen_pairs = set()
    for p_ in P.PLAN:
        card = p_.get("gpu") or mv2.GPU
        for m in p_.get("models", []):
            key = next((k for k in sorted(BILLIONS, key=len, reverse=True)
                        if k in m.lower()), None)
            if key and (m, card) not in seen_pairs:
                seen_pairs.add((m, card))
                print(f"         {m:28s} ~{BILLIONS[key]*2:4.0f} GB on {card}")

    for name, f in (("phase1 probe", lambda i: i["n_turns"] == 4),
                    ("phase2 probe",
                     lambda i: i["condition"].split("_")[0] in
                     {"T0", "T1", "T2"})):
        b = len(json.dumps([i for i in items if f(i)]).encode())
        (ok if b < 100e6 else fail)(
            f"{name}: items payload {b/1e6:.2f} MB per container "
            f"({5*b/1e6:.1f} MB across 5 models)")


def check_plan(items):
    head("PLAN -- the emitted commands are runnable and do not collide")
    sys.argv = ["preflight"]
    import run_plan as P
    if not os.path.exists(P.RUNNER):
        fail(f"run_plan points at {P.RUNNER}, which does not exist")
    else:
        ok(f"run_plan points at {P.RUNNER}")

    tags = [p.get("tag") for p in P.PLAN if p["exp"] != "ANALYSIS-ONLY"]
    if any(t is None for t in tags):
        fail("some plan rows have no tag")
    dup = [t for t, c in Counter(tags).items() if c > 1]
    if dup:
        fail(f"duplicate tags {dup}: those runs overwrite each other's CSV")
    else:
        ok(f"{len(tags)} runs, {len(set(tags))} distinct output tags")

    # the models a run names must be able to appear in a filter
    for p in P.PLAN:
        if p["exp"] == "ANALYSIS-ONLY":
            continue
        want = {c.strip() for c in p["conds"].split(",")}
        turns = {int(t) for t in p["turns"].split(",")}
        poss = {float(x) for x in p.get("pos", P.REF_POS).split(",")}
        frms = {f.strip() for f in p.get("frames", P.REF_FRAME).split(",")}
        sub = [i for i in items
               if (i["condition"] in want
                   or i["condition"].split("_")[0] in want)
               and i["n_turns"] in turns
               and float(i["event_pos"]) in poss
               and i.get("setup_frame", "requirement") in frms
               and (not p.get("source") or i["source"] == p["source"])
               and (p.get("realization") is None
                    or i["realization"] == p["realization"])]
        if not sub:
            fail(f"{p['name']}: no items match its filters")
            continue
        n_pred, _ = P.cost(p)
        n_real = len(sub) * P.FWD_PER_ITEM * len(p["models"])
        if p.get("max_items"):
            n_real = min(len(sub), p["max_items"]) * 2 * len(p["models"])
        if p["exp"] == "probecheck":
            n_real = min(len(sub), p.get("max_items", 40)) * 4 * len(p["models"])
        if abs(n_pred - n_real) > max(20, 0.05 * n_real):
            warn(f"{p['name']}: plan predicts {n_pred:,} passes, the filters "
                 f"give {n_real:,}")
        else:
            ok(f"{p['name']}: {n_real:,} passes over {len(sub)} items")

    out = "results_v2"
    if os.path.isdir(out):
        have = [f for f in os.listdir(out) if f.endswith(".csv")]
        if have:
            warn(f"{out}/ already holds {have}. The runner now refuses to "
                 "overwrite, so move them aside or use a new --tag.")


def check_human_baseline():
    head("GROUND TRUTH -- where the answer key comes from")
    import glob
    d = json.load(open(ANCHORS))
    ev = [e for a in d["attributes"] for e in a["events"]]
    org = Counter(e.get("origin", "?") for e in ev)
    ok(f"{len(ev)} causal claims, all marked source=author (validated); "
       f"first drafted by: {dict(org)}")
    print("       The 50 author-written events match data/anchors/anchors_template_filled.json")
    print("       exactly (one documented rename). The 27 LLM-drafted ones plus")
    print("       replytone/replyformat were added to close structural gaps, then")
    print("       endorsed. Chain is anchors_full.json -> data/items_v2.json, never the")
    print("       reverse.")

    v2 = {i["id"] for i in json.load(open(ITEMS))}
    best = None
    for d in ("data/annotation", "data/anchors"):
        key = os.path.join(d, "_answer_key.json")
        if not os.path.exists(key):
            continue
        ov = len(set(json.load(open(key))) & v2)
        done = total = 0
        for f in glob.glob(os.path.join(d, "annotator_*.json")):
            its = json.load(open(f)).get("items", [])
            total += len(its)
            done += sum(1 for i in its
                        if str(i.get("answer", "TODO")).strip() != "TODO")
        # The annotation was returned as a spreadsheet, so a completed form
        # lives in a *_filled.csv and never reaches the JSON. Counting only
        # the JSON reported zero judgements on a set that was in fact done.
        for f in glob.glob(os.path.join(d, "annotator_*_filled.csv")):
            rows = list(csv.DictReader(open(f)))
            total += len(rows)
            done += sum(1 for r in rows
                        if str(r.get("answer", "")).strip() not in ("", "TODO"))
        if best is None or ov > best[1]:
            best = (d, ov, done, total)
    if best is None or best[1] == 0:
        warn("No annotation set matches these items. Build one: "
             "python src/make_annotation.py build --out data/annotation")
    else:
        d, ov, done, total = best
        ok(f"{d}/ targets {ov} ids that exist in {ITEMS}")
        if done == 0:
            warn(f"{d}/ is built but unanswered (0/{total} judgements). Until "
                 "it is filled in there is no human ceiling for T2, and a "
                 "2-11% model result invites the objection that the items are "
                 "unanswerable. Author validation of the causal claims does "
                 "not cover this: it does not show that a reader who has NOT "
                 "seen the anchors recovers the same answer.")
        elif done < total:
            warn(f"{d}/ partially answered ({done}/{total} judgements). "
                 "Score it with: python src/make_annotation.py score --dir " + d)
        else:
            ok(f"{d}/ complete ({done}/{total} judgements)")


# fields whose absence cannot be repaired without re-running the GPU
MUST_RETAIN = {
    "run_probe": [
        ("opt_a", "which two options were displayed -- without it `raw` and "
                  "`picked` cannot be re-scored at all"),
        ("p_correct", "probability on the CORRECT option; needed for "
                      "calibration and confidence-weighted accuracy. v1 had "
                      "it, v2 dropped it, and nobody noticed until a "
                      "below-chance result made 'how confidently wrong' the "
                      "interesting question"),
        ("logit_margin", "how close the call was"),
    ],
    "run_generate": [
        ("opt_a", "the displayed pair; without it a parser fix means a re-run"),
        ("n_raw_chars", "the TRUE length before storing, so clipping is "
                        "detectable after the fact"),
        ("raw_clipped", "whether this row's text can be re-scored"),
    ],
}


def check_retention(items):
    head("RETENTION -- can a run's CSV still answer the run's question")
    src = open(RUNNER).read()

    for fn, fields in MUST_RETAIN.items():
        body = src[src.index(f"def {fn}("):]
        body = body[:body.index("\n@app.function") if "\n@app.function" in body
                    else len(body)]
        for name, why in fields:
            if f"{name}=" in body or f'"{name}"' in body:
                ok(f"{fn} records `{name}`")
            else:
                fail(f"{fn} does not record `{name}` -- {why}")

    # head-only clipping destroys the end of a reply, which is where the
    # answer is. This is the exact bug that cost a phase-3 trace analysis.
    # Walk the AST -- a string search also hits the comment explaining the bug.
    import ast as _ast
    bad_slices = []
    for node in _ast.walk(_ast.parse(src)):
        if not isinstance(node, _ast.keyword) or node.arg not in ("raw",
                                                                  "reasoning"):
            continue
        v = node.value
        if not (isinstance(v, _ast.Subscript)
                and isinstance(v.slice, _ast.Slice)):
            continue
        # x[:N] keeps the head and throws the tail away. Only the generated
        # text matters here -- `raw=str(e)[:300]` on the load-failure path is
        # an error message, not a reply.
        target = v.value
        if not (isinstance(target, _ast.Name)
                and target.id in ("text", "reasoning", "visible")):
            continue
        if v.slice.lower is None and v.slice.upper is not None:
            bad_slices.append(f"{node.arg}={target.id}[:N]")
    if bad_slices:
        fail(f"the runner stores {sorted(set(bad_slices))} head-first. "
             "The answer tag is at the END of a reply, so a clipped "
             "record cannot be re-scored. Use _clip(), which keeps head AND "
             "tail.")
    elif "_clip(" in src:
        ok("raw/reasoning stored head+tail via _clip(), so a clipped row is "
           "still re-scorable near the answer")

    # is the configured budget big enough for the traces we have actually seen?
    import glob
    seen = []
    for f in sorted(glob.glob("results/*.csv")):
        try:
            import pandas as pd
            d = pd.read_csv(f, usecols=lambda c: c in
                            ("n_reason_chars", "raw_clipped", "thinking"))
        except Exception:
            continue
        if "n_reason_chars" not in d or not len(d):
            continue
        p95 = float(d.n_reason_chars.quantile(.95))
        seen.append((os.path.basename(f), p95, len(d),
                     float(d.raw_clipped.mean()) if "raw_clipped" in d else None))
    m = re.search(r"raw_chars: int = (\d+)", src)
    budget = int(m.group(1)) if m else None
    if seen:
        print(f"       trace lengths observed in results/ "
              f"(configured --raw-chars default: {budget}):")
        for name, p95, n, cl in seen:
            note = "" if cl is None else f", {100*cl:.0f}% flagged clipped"
            print(f"         {name}: n={n}, p95 reasoning "
                  f"{p95:.0f} chars{note}")
            if budget and p95 > budget:
                warn(f"{name} shows p95 reasoning of {p95:.0f} chars but "
                     f"--raw-chars defaults to {budget}. A reasoning run at "
                     f"this default loses the tail on the longest traces; "
                     f"pass --raw-chars {int(p95*1.3//1000+1)*1000} for any "
                     "run whose traces you intend to analyse.")
    else:
        print("       (no run yet records n_reason_chars; nothing to calibrate "
              "the budget against)")

    # what each existing CSV can and cannot answer
    import glob as _g
    for f in sorted(_g.glob("results/*.csv")):
        if "_rescored" in f or "_judged" in f:
            continue
        try:
            import pandas as pd
            cols = set(pd.read_csv(f, nrows=1).columns)
        except Exception:
            continue
        if "probecheck" in os.path.basename(f):
            ok(f"{os.path.basename(f)} is a format diagnostic; ab_mass is all "
               "it needs to retain")
            continue
        miss = []
        if "p_correct" not in cols and "ab_mass" in cols:
            miss.append("calibration (no p_correct)")
        if "opt_a" not in cols:
            miss.append("any re-parse (no displayed pair)")
        if "raw" in cols and "raw_clipped" not in cols:
            miss.append("trace analysis (raw clipped, extent unrecorded)")
        if miss:
            warn(f"{os.path.basename(f)} cannot support: {'; '.join(miss)}. "
                 "Re-running is the only fix.")
        else:
            ok(f"{os.path.basename(f)} retains what its experiment needs")


def _by(items, k):
    d = defaultdict(list)
    for i in items:
        d[i[k]].append(i)
    return d


def main():
    check_env()
    items, attrs = check_data()
    check_design(items)
    check_leakage(items)
    check_rendering(items)
    check_resources(items)
    check_plan(items)
    check_retention(items)
    check_human_baseline()

    print("\n" + "=" * 70)
    if FAIL:
        print(f"{len(FAIL)} BLOCKER(S) -- do not launch:")
        for f in FAIL:
            print(f"  - {f}")
    if WARN:
        print(f"\n{len(WARN)} warning(s) -- judgement calls, not blockers:")
        for w in WARN:
            print(f"  - {w}")
    if not FAIL:
        print("\npreflight clear. Next: python test_parsing.py, then phase 0.")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
