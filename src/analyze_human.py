#!/usr/bin/env python3
"""Score the human annotation and compare it with the models item by item.

    python analyze_human.py

The unconditional accuracy answers "can a person do this task". The accuracy conditioned on the
annotator's own `old_ok` field answers "can a person do this task on the items
where they accept that the event makes the stated preference unusable". The
second is the ceiling the paper quotes, because an annotator who rejects an
item's premise is not failing to infer, they are disagreeing with the key, and
scoring that as a human error understates the ceiling.

The same split is applied to the models on exactly those items, so the human
and the model numbers are never computed over different sets.
"""
import json
import math

import pandas as pd

FORM = "data/annotation/annotator_A_filled.csv"
KEY = "data/annotation/_answer_key.json"
MODEL_RUNS = ("probe_chat_prefill__phase1_main",
              "generate_thinkon__phase3_thinkon",
              "generate_thinkoff__phase5_gen_req_rescored")
CONDS = ["T0_no_evidence", "T1_explicit", "T2_implicit",
         "T3_counterfact", "T4_conflict"]
# Language the annotator used to argue that a narrow event does not override
# a standing preference. Counted in traces to test whether the models reach
# the same answer by the same route.
SCOPE = ["temporar", "one-off", "one off", "for now", "short term",
         "short-term", "long run", "long-term", "permanent", "occasional",
         "just this", "only for this", "situational",
         "depends on the situation", "still fine", "does not mean",
         "doesn't mean", "can still"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def load():
    h = pd.read_csv(FORM)
    key = json.load(open(KEY))
    for f in ("condition", "expected", "old", "event_force"):
        h[f] = h.item_id.map(lambda i: key[i].get(f, ""))
    for f in ("old_ok", "best_pick"):
        h[f] = h[f].astype(str).str.strip().str.lower()
    h["ok"] = h.answer.str.lower().str.strip() == h.expected.str.lower()
    h["picked_old"] = h.answer.str.lower().str.strip() == h.old.str.lower()
    parts = []
    for f in MODEL_RUNS:
        d = pd.read_csv(f"results/{f}.csv", low_memory=False)
        if "parsed" in d:
            d = d[d.parsed.fillna(True).astype(bool)]
        parts.append(d[["id", "is_correct"]])
    m = pd.concat(parts, ignore_index=True)
    m["is_correct"] = m.is_correct.astype(bool)
    return h, m


def main():
    h, m = load()
    print(f"annotator A, {len(h)} items, {h.answer.notna().sum()} answered\n")
    print(f"{'condition':16s} {'n':>3s} {'human':>7s}  95% CI")
    for c in CONDS:
        s = h[h.condition == c]
        a = wilson(int(s.ok.sum()), len(s))
        print(f"{c:16s} {len(s):3d} {a[0]:6.1f}%  [{a[1]:.1f}, {a[2]:.1f}]")

    print("\nsplit on the annotator's own premise check (old_ok), with the "
          "models scored on exactly the same items")
    print(f"{'condition':14s} {'premise':10s} {'items':>5s} {'human':>7s} "
          f"{'models':>8s}")
    for c in ("T2_implicit", "T4_conflict"):
        for lab, flag in (("accepted", "no"), ("disputed", "yes")):
            hs = h[(h.condition == c) & (h.old_ok == flag)]
            ids = set(hs.item_id) | {i.replace("_finc", "") for i in hs.item_id}
            ms = m[m.id.isin(ids)]
            ha = wilson(int(hs.ok.sum()), len(hs))
            ma = wilson(int(ms.is_correct.sum()), len(ms))
            print(f"{c:14s} {lab:10s} {len(hs):5d} {ha[0]:6.1f}% "
                  f"{ma[0]:7.1f}%  [{ma[1]:.1f}, {ma[2]:.1f}] over {len(ms)}")

    print("\nboth gates: the annotator says the stated option no longer "
          "works AND the keyed option is the clearly better suggestion")
    print(f"{'condition':14s} {'gate':32s} {'items':>5s} {'human':>7s} "
          f"{'models':>8s}")
    gates = (("all items", lambda d: d.index == d.index),
             ("premise accepted", lambda d: d.old_ok == "no"),
             ("premise accepted + clearly better",
              lambda d: (d.old_ok == "no") & (d.best_pick == "yes")))
    for c in ("T2_implicit", "T4_conflict"):
        for lab, g in gates:
            hs = h[(h.condition == c)]
            hs = hs[g(hs)]
            ids = set(hs.item_id) | {i.replace("_finc", "") for i in hs.item_id}
            ms = m[m.id.isin(ids)]
            ha = wilson(int(hs.ok.sum()), len(hs))
            ma = wilson(int(ms.is_correct.sum()), len(ms))
            print(f"{c:14s} {lab:32s} {len(hs):5d} {ha[0]:6.1f}% "
                  f"{ma[0]:7.1f}%  [{ma[1]:.1f}, {ma[2]:.1f}] over {len(ms)}")

    print("\nwhen the annotator disagreed with the key, what did they choose?")
    for c in ("T2_implicit", "T4_conflict"):
        s = h[(h.condition == c) & (~h.ok)]
        print(f"  {c:14s} {len(s):2d} disagreements, "
              f"{int(s.picked_old.sum())} chose the stated preference, "
              f"{int((s.old_ok != 'no').sum())} had marked it still workable")

    print("\ndoes the model give the annotator's reason? scope-and-duration "
          "language in traces")
    d = pd.read_csv("results/generate_thinkon__phase3_thinkon.csv",
                    low_memory=False)
    d = d[d.parsed].copy()
    low = d.raw.astype(str).str.lower()
    d["scope"] = low.apply(lambda s: any(w in s for w in SCOPE))
    for c in ("T2_implicit", "T4_conflict"):
        s = d[d.condition == c]
        print(f"  {c:14s} {100*s.scope.mean():5.1f}% of traces, "
              f"accuracy {100*s[s.scope].is_correct.mean():.1f}% with it "
              f"and {100*s[~s.scope].is_correct.mean():.1f}% without")
    notes = h[h.notes.notna() & (h.condition == "T4_conflict")]
    n_scope = sum(any(w in str(x).lower() for w in SCOPE) or "but" in str(x).lower()
                  for x in notes.notes)
    print(f"  annotator, T4 notes: {n_scope} of {len(notes)} make that argument")


if __name__ == "__main__":
    main()
