#!/usr/bin/env python3
"""
Human annotation for the event-to-preference items.

    python make_annotation.py build --out annotation_v2
    # ... annotators fill data/annotation/annotator_A_form.csv in a spreadsheet
    python make_annotation.py ingest --dir annotation_v2
    python make_annotation.py score  --dir annotation_v2

WHY
---
Every item assumes a person reading the conversation would infer the preference
change. That assumption has never been tested on the v2 set. Phase 1 put T2 at
2-11% and T4 at 1-6% for every model, i.e. far BELOW chance -- the models keep
the preference stated in the setup. A number that extreme invites exactly one
objection: the items are unanswerable. Only human data settles it.

The author wrote and endorsed all 77 causal claims, which establishes that the
claims are sound. It does not establish that a reader who has NOT seen the
anchors recovers the same answer from the conversation alone. That is the gap
this fills, and it is the difference between "models fail a task people do
easily" and "models fail an ambiguous task".

WHAT THE SAMPLE IS WEIGHTED FOR
-------------------------------
  T2_implicit    the headline. Weighted heaviest.
  T4 pull/push   both, deliberately. The push/pull split (see
                 anchors_full.json `force`) is an authored judgement about
                 whether T4's event still bears on the swapped-in setup value.
                 If annotators keep the setup value on push items and switch on
                 pull ones, that split is confirmed by data rather than assertion.
  T1_explicit    attention check. A reader who misses these is not reading.
  T0_no_evidence floor check. Nothing happened, so the setup value is right.
  T3_counterfact the setup value is not on offer at all.

Items come from the SAME reference cell the models saw (4 turns, event_pos 0.6)
and are rendered with the same builder, so the comparison is like for like.

WHAT THE ANNOTATOR SEES
-----------------------
The conversation as plain text, the two options, and three fields to fill in.
No model outputs, no correct answers, no hints about which option we expect.
Option order is randomised per item and per annotator.
"""

import argparse, json, os, random, sys, importlib.util
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
# The repo root, one level above src/. Data paths are resolved from there.
ROOT = os.path.dirname(HERE)


def _load_v2(items_file, turns, pos):
    """The v2 items, rendered by the same builder the runner uses.

    This used to string-slice modal_behavioral.py -- the v1 runner -- so the
    annotation set had zero overlap with data/items_v2.json and could not serve as a
    ceiling for anything actually run.
    """
    sys.argv = [sys.argv[0]]                    # generate_v2 parses argv
    import generate_v2 as G
    items = [i for i in json.load(open(items_file))
             if i["n_turns"] == turns and abs(i["event_pos"] - pos) < 1e-9]
    if not items:
        sys.exit(f"no items at n_turns={turns}, event_pos={pos}")
    return G, items


def _render(G, it):
    """Conversation only -- no question, no options, header stripped."""
    convo, _ = G.build_conversation(it, it["n_turns"], it["event_pos"],
                                    random.Random(it["seed"]))
    return convo.replace(G.HEADER, "").rstrip()


# How many of each stratum every annotator sees. Sized for TWO annotators with
# full overlap: 60 items, 120 judgements, pairwise agreement on all 60.
#
# The precision of the human ceiling comes from the number of ITEMS, not
# judgements -- two ratings of the same item are correlated, so they tighten
# agreement rather than the ceiling. With one fewer annotator the budget is
# therefore spent on more T2 items, which is the condition the paper turns on:
# 24 items puts a 21/24 result at roughly [69,96], decisive against a model
# sitting at 5%.
QUOTA = [
    # Rebalanced for RESPONSE BIAS, not just for T2 precision. The first
    # version was 24 T2 / 4 T0, which keyed 93% of items as "switch" -- an
    # annotator who noticed that could score high on T2 by defaulting to
    # switch whenever an event appeared, without making the judgement the
    # ceiling is supposed to measure. That would inflate the one number the
    # paper's central comparison rests on.
    #
    # T0 carries no event, so staying is unambiguously correct there.
    # T4-push carries an event that does not bear on the stated preference,
    # so staying is defensible with an event present -- which is the case
    # that actually breaks the bias, since the annotator must still judge.
    #
    # Cost: 20 T2 items instead of 24. A 17/20 result gives roughly [64,95],
    # still decisive against a model at 5%.
    ("T2_implicit",    None,   20),   # the headline
    ("T0_no_evidence", None,   12),   # no event -> stay is correct
    ("T4_conflict",    "push", 12),   # event present, stay defensible
    ("T4_conflict",    "pull", 10),   # the sound half of T4
    ("T1_explicit",    None,    4),   # attention check, expect ~100%
    ("T3_counterfact", None,    2),   # spot-check only, not powered
]

FORM_COLS = ["item_id", "conversation", "question", "option_1", "option_2",
             "answer", "confidence", "old_ok", "best_pick", "notes"]


INSTRUCTIONS = [
    "You are reading short excerpts from someone's past conversations with an assistant.",
    "",
    "For each item, fill in the three fields marked TODO. Do not change anything else.",
    "",
    "  answer      -> type one of the two options exactly as written.",
    "                 Pick what an assistant SHOULD SUGGEST to this person now.",
    "                 Not what they secretly prefer -- what you would suggest.",
    "  confidence  -> 1 (guessing) to 5 (certain).",
    "  old_ok      -> yes / no / unclear.",
    "                 Would it still be APPROPRIATE to suggest the person's",
    "                 original choice to them now?",
    "                 This is about what you should suggest, not about what they",
    "                 secretly like. Someone can still love running and it still",
    "                 be wrong to suggest it after a knee injury.",
    "",
    "  best_pick   -> yes / no / unclear.",
    "                 Of the two options offered, is your answer clearly the",
    "                 better thing to suggest?",
    "                 'no' if both are bad, or if you are only picking it because",
    "                 the other one is ruled out. That distinction matters and we",
    "                 want to know about it.",
    "",
    "  notes       -> optional, free text. Especially useful for 'no'/'unclear'.",
    "",
    "Answer from the conversation alone. There is no trick and no time limit.",
    "Some items may be genuinely ambiguous; saying so is a useful result.",
]


def build(out_dir, annotators, seed, items_file, turns, pos):
    G, pool = _load_v2(items_file, turns, pos)
    leaks = G.check_leaks(G.load_anchors(os.path.join(
        ROOT, "data", "anchors", "anchors_full.json")), verbose=False)
    assert not leaks, f"fix these first: {leaks}"

    # One sampled set, shown to every annotator. Full overlap is deliberate:
    # a per-annotator sample gives no inter-annotator agreement number, and
    # agreement is the thing that says whether an item is stable.
    rng = random.Random(seed)
    chosen, used = [], set()
    for cond, force, n in QUOTA:
        cand = [i for i in pool if i["condition"] == cond
                and (force is None or i.get("event_force") == force)
                and i["id"] not in used]
        rng.shuffle(cand)
        # Round-robin over attributes so one attribute cannot fill a stratum:
        # take at most one per attribute per sweep, sweeping until the quota is
        # met or the candidates run out.
        buckets = defaultdict(list)
        for i in cand:
            buckets[i["attribute"]].append(i)
        attrs = sorted(buckets)
        rng.shuffle(attrs)
        take = []
        while len(take) < n and any(buckets[a] for a in attrs):
            for a in attrs:
                if len(take) >= n:
                    break
                if buckets[a]:
                    take.append(buckets[a].pop())
        if len(take) < n:
            print(f"  ! {cond}/{force}: wanted {n}, only {len(take)} available")
        used.update(i["id"] for i in take)
        chosen.extend(take)
    os.makedirs(out_dir, exist_ok=True)

    key = {it["id"]: dict(attribute=it["attribute"], condition=it["condition"],
                          family=it["family"], event_force=it.get("event_force", ""),
                          event_id=it["event_id"], realization=it["realization"],
                          expected=it["correct"], other=it["distractor"],
                          old=it["old"], new=it["new"],
                          event=it["event_text"], setup=it["setup_text"])
           for it in chosen}
    with open(os.path.join(out_dir, "_answer_key.json"), "w") as f:
        json.dump(key, f, indent=2)

    for a in annotators:
        arng = random.Random(f"{a}-{seed}".__hash__() & 0x7fffffff)
        order = chosen[:]
        arng.shuffle(order)
        payload = []
        for it in order:
            opts = [it["correct"], it["distractor"]]
            arng.shuffle(opts)                      # per annotator, per item
            payload.append({
                "item_id": it["id"],
                "conversation": _render(G, it),
                "question": (f"{it['name']} asks you for a suggestion about "
                             f"their {it['noun']}. Which should you suggest?"),
                "options": opts,
                "answer": "TODO",
                "confidence": "TODO",
                "old_ok": "TODO",
                "best_pick": "TODO",
                "notes": "",
            })
        doc = {"annotator": a, "instructions": INSTRUCTIONS,
               "n_items": len(payload), "items": payload}
        p = os.path.join(out_dir, f"annotator_{a}.json")
        with open(p, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"wrote {p}  ({len(payload)} items)")

    from collections import Counter as _C
    print(f"\n{len(chosen)} items, same set for all {len(annotators)} annotators "
          f"({len(chosen)*len(annotators)} judgements, every item multiply rated)")
    print("  condition :", dict(_C(i["condition"] for i in chosen)))
    print("  T4 force  :", dict(_C(i.get("event_force") for i in chosen
                                   if i["condition"] == "T4_conflict")))
    print("  attributes:", len({i["attribute"] for i in chosen}), "of 12")
    print("  family    :", dict(_C(i["family"] for i in chosen)))
    stay = [i for i in chosen if i["correct"] == i["old"]]
    push = [i for i in chosen if i["condition"] == "T4_conflict"
            and i.get("event_force") == "push"]
    n = len(chosen)
    print(f"\n  RESPONSE BIAS CHECK")
    print(f"    keyed as STAY with the stated value : {len(stay):3d}/{n} "
          f"= {100*len(stay)/n:.0f}%")
    print(f"    keyed as SWITCH                     : {n-len(stay):3d}/{n} "
          f"= {100*(n-len(stay))/n:.0f}%")
    print(f"    defensible-to-stay (incl. T4 push)  : "
          f"{len(stay)+len(push):3d}/{n} = {100*(len(stay)+len(push))/n:.0f}%")
    if len(stay) / n < 0.15:
        print("    >>> under 15% keyed as stay. An annotator who notices that")
        print("        almost every item involves a change can score high on")
        print("        T2 without judging. Raise the T0 quota.")

    # --- spreadsheet forms -------------------------------------------------
    # Hand-editing 45 KB of JSON invites exactly the malformed rows the scorer
    # rejects. One row per item, the conversation in a quoted multi-line cell
    # (Excel and Sheets both read these), and the four fields to fill as the
    # last columns.
    import csv as _csv
    for aa in annotators:
        doc = json.load(open(os.path.join(out_dir, f"annotator_{aa}.json")))
        fp = os.path.join(out_dir, f"annotator_{aa}_form.csv")
        with open(fp, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=FORM_COLS)
            w.writeheader()
            for i in doc["items"]:
                w.writerow({"item_id": i["item_id"],
                            "conversation": i["conversation"],
                            "question": i["question"],
                            "option_1": i["options"][0],
                            "option_2": i["options"][1],
                            "answer": "", "confidence": "",
                            "old_ok": "", "best_pick": "", "notes": ""})
        print(f"wrote {fp}  ({len(doc['items'])} rows)")

    ip = os.path.join(out_dir, "INSTRUCTIONS.txt")
    with open(ip, "w") as f:
        f.write("\n".join(INSTRUCTIONS) + "\n\n" + "=" * 70 + "\n\n")
        f.write("HOW TO FILL THE FORM\n\n"
                "Open annotator_<your letter>_form.csv in Excel or Google\n"
                "Sheets. One row per item. Read the `conversation` cell, then\n"
                "fill the last four columns. Leave every other column alone.\n\n"
                "  answer      one of the two words in option_1 / option_2,\n"
                "              typed exactly\n"
                "  confidence  1 to 5\n"
                "  old_ok      yes / no / unclear\n"
                "  best_pick   yes / no / unclear\n"
                "  notes       optional\n\n"
                "Save as CSV, keeping the same filename. Answer from the\n"
                "conversation alone. Some items may be genuinely ambiguous --\n"
                "saying so through old_ok and best_pick is a useful result,\n"
                "not a failure.\n")
    print(f"wrote {ip}")

    txt = os.path.join(out_dir, "readable_preview.txt")
    with open(txt, "w") as f:
        f.write("\n".join(INSTRUCTIONS) + "\n\n" + "=" * 70 + "\n")
        for r in payload[:5]:
            f.write(f"\n[{r['item_id']}]\n{r['conversation']}\n\n"
                    f"{r['question']}\n"
                    f"  ( ) {r['options'][0]}\n  ( ) {r['options'][1]}\n"
                    f"  confidence 1-5: ____  old still ok? ___  best pick? ___\n"
                    f"  notes: ______________________________________\n"
                    + "-" * 70 + "\n")
    print(f"wrote {txt}  (first 5 items, for eyeballing)")
    print(f"\nanswer key at {out_dir}/_answer_key.json -- do not send this out")


def ingest(dir_):
    """Read the filled CSV forms back into annotator_*.json.

    The JSON stays the canonical input to `score`, so the spreadsheet is only a
    data-entry surface. Nothing is written unless every row of a file parses:
    a half-ingested file would score a subset and quietly bias the ceiling.
    """
    import csv as _csv
    forms = sorted(f for f in os.listdir(dir_)
                   if f.startswith("annotator_") and f.endswith("_form.csv"))
    if not forms:
        sys.exit(f"no annotator_*_form.csv in {dir_}")
    for fn in forms:
        a = fn[len("annotator_"):-len("_form.csv")]
        jp = os.path.join(dir_, f"annotator_{a}.json")
        if not os.path.exists(jp):
            print(f"{fn}: no annotator_{a}.json to merge into, skipping")
            continue
        doc = json.load(open(jp))
        by_id = {i["item_id"]: i for i in doc["items"]}
        rows = list(_csv.DictReader(open(os.path.join(dir_, fn))))
        problems, filled = [], 0
        staged = {}
        for n, r in enumerate(rows, start=2):          # row 1 is the header
            iid = (r.get("item_id") or "").strip()
            if iid not in by_id:
                problems.append(f"row {n}: unknown item_id {iid!r}")
                continue
            ans = (r.get("answer") or "").strip()
            if not ans:
                continue                               # simply not done yet
            opts = [o.lower() for o in by_id[iid]["options"]]
            conf = (r.get("confidence") or "").strip()
            ok = (r.get("old_ok") or "").strip().lower()
            bp = (r.get("best_pick") or "").strip().lower()
            if ans.lower() not in opts:
                problems.append(f"row {n} ({iid}): answer {ans!r} is not one "
                                f"of {by_id[iid]['options']}")
            if conf not in {"1", "2", "3", "4", "5"}:
                problems.append(f"row {n} ({iid}): confidence {conf!r} "
                                "is not 1-5")
            if ok not in {"yes", "no", "unclear"}:
                problems.append(f"row {n} ({iid}): old_ok {ok!r} is not "
                                "yes/no/unclear")
            if bp not in {"yes", "no", "unclear"}:
                problems.append(f"row {n} ({iid}): best_pick {bp!r} is not "
                                "yes/no/unclear")
            staged[iid] = dict(answer=ans, confidence=conf, old_ok=ok,
                               best_pick=bp,
                               notes=(r.get("notes") or "").strip())
            filled += 1
        if problems:
            print(f"\n{fn}: {len(problems)} problem(s) -- NOTHING written for "
                  "this annotator:")
            for msg in problems[:12]:
                print(f"    {msg}")
            if len(problems) > 12:
                print(f"    ... and {len(problems)-12} more")
            print("  Fix the form and re-run. A partial merge would score a")
            print("  subset of the items and bias the ceiling.")
            continue
        for iid, v in staged.items():
            by_id[iid].update(v)
        json.dump(doc, open(jp, "w"), indent=2, ensure_ascii=False)
        print(f"{fn}: merged {filled}/{len(rows)} completed rows -> "
              f"annotator_{a}.json")
    print("\nNow: python make_annotation.py score --dir " + dir_)


def score(dir_):
    key = json.load(open(os.path.join(dir_, "_answer_key.json")))
    files = [f for f in os.listdir(dir_)
             if f.startswith("annotator_") and f.endswith(".json")]
    if not files:
        sys.exit(f"no annotator_*.json in {dir_}")

    per_item = defaultdict(list)
    rows = []
    for fn in sorted(files):
        doc = json.load(open(os.path.join(dir_, fn)))
        a = doc["annotator"]
        done = [i for i in doc["items"] if str(i.get("answer", "TODO")).strip() != "TODO"]
        if not done:
            print(f"{fn}: not started, skipping")
            continue

        # validate before scoring: silently dropping malformed rows biases the
        # baseline toward whatever the careful annotators did
        problems = []
        for i in done:
            opts = [o.lower() for o in i["options"]]
            ans = str(i.get("answer", "")).strip().lower()
            conf = str(i.get("confidence", "")).strip()
            chg = str(i.get("old_ok", "")).strip().lower()
            bp = str(i.get("best_pick", "")).strip().lower()
            if ans not in opts:
                problems.append((i["item_id"], f"answer {ans!r} is not one of {opts}"))
            if conf not in {"1", "2", "3", "4", "5"}:
                problems.append((i["item_id"], f"confidence {conf!r} is not 1-5"))
            if chg not in {"yes", "no", "unclear"}:
                problems.append((i["item_id"], f"old_ok {chg!r} is not yes/no/unclear"))
            if bp not in {"yes", "no", "unclear"}:
                problems.append((i["item_id"], f"best_pick {bp!r} is not yes/no/unclear"))
        if problems:
            print(f"  {len(problems)} field problems in {fn}:")
            for iid, msg in problems[:10]:
                print(f"    {iid}: {msg}")
            if len(problems) > 10:
                print(f"    ... and {len(problems)-10} more")
            print("  >>> fix these and re-run; they are excluded for now")
        # contradiction: "the old option is not appropriate" but the answer IS
        # the old option. One rater did this on tone_rev, reading the setup as a
        # standing request the event does not override. Flag rather than score.
        contra = []
        for i in done:
            k = key[i["item_id"]]
            ans = str(i.get("answer", "")).strip().lower()
            if str(i.get("old_ok", "")).strip().lower() == "no" and \
                    ans == k["other"].lower():
                contra.append(i["item_id"])
        if contra:
            print(f"  {len(contra)} CONTRADICTORY judgements (old_ok=no but the")
            print(f"  answer is the old option): {contra[:6]}")
            print("  These need re-reading, not scoring.")
        problems += [(c, "old_ok=no but answered the old option") for c in contra]

        bad_ids = {p[0] for p in problems}
        done = [i for i in done if i["item_id"] not in bad_ids]
        if not done:
            print(f"{fn}: nothing usable after validation"); continue

        for i in done:
            k = key[i["item_id"]]
            correct = i["answer"].strip().lower() == k["expected"].lower()
            rows.append(dict(annotator=a, item=i["item_id"],
                             domain=k["attribute"],
                             direction=k["condition"],
                             force=k.get("event_force", ""), correct=correct,
                             confidence=i["confidence"],
                             old_ok=str(i.get("old_ok", "")).strip().lower(),
                             best_pick=str(i.get("best_pick", "")).strip().lower(),
                             answer=i["answer"].strip().lower()))
            per_item[i["item_id"]].append(i["answer"].strip().lower())
        print(f"{fn}: {len(done)}/{len(doc['items'])} completed")

    import statistics as st
    n = len(rows)
    if not n:
        print("\nNothing to score yet: every `answer` field is still TODO.")
        print("Fill in annotator_*.json, then re-run. Each item needs four")
        print("fields: answer, confidence (1-5), old_ok, best_pick.")
        return
    acc = 100 * sum(r["correct"] for r in rows) / n
    print(f"\n=== HUMAN BASELINE ===\nn = {n} judgements from "
          f"{len({r['annotator'] for r in rows})} annotators")
    print(f"agreement with the intended answer: {acc:.1f}%")

    print(f"old still appropriate?: {dict(Counter(r['old_ok'] for r in rows))}")
    print(f"answer is clearly best?: {dict(Counter(r['best_pick'] for r in rows))}")
    elim = [r for r in rows if r["old_ok"] == "no" and r["best_pick"] != "yes"]
    if elim:
        print(f"\n  {len(elim)} judgements ({100*len(elim)/len(rows):.0f}%) where the")
        print("  original is ruled out but the replacement is not clearly right.")
        print("  Those items measure ELIMINATION, not inference. Report the")
        print("  appropriateness question for them rather than the forced choice.")

    print("\n--- by attribute ---")
    bydom = defaultdict(list)
    for r in rows:
        bydom[r["domain"]].append(r)
    print(f"{'attribute':14s} {'n':>4s} {'agree %':>8s} {'best_pick=yes %':>16s}")
    for dom, rs in sorted(bydom.items(),
                          key=lambda kv: -sum(x["correct"] for x in kv[1]) / len(kv[1])):
        a = 100 * sum(x["correct"] for x in rs) / len(rs)
        y = 100 * sum(x["best_pick"] == "yes" for x in rs) / len(rs)
        flag = "   <-- consider dropping" if a < 70 or y < 60 else ""
        print(f"{dom:14s} {len(rs):4d} {a:8.1f} {y:16.1f}{flag}")

    print("\n--- by condition: THE human ceiling, per condition ---")
    print(f"{'condition':16s} {'n':>4s} {'agree %':>8s} {'best_pick=yes':>14s}"
          f" {'mean conf':>10s}")
    for d in ("T0_no_evidence", "T1_explicit", "T2_implicit",
              "T3_counterfact", "T4_conflict"):
        rs = [r for r in rows if r["direction"] == d]
        if not rs:
            continue
        a = 100 * sum(x["correct"] for x in rs) / len(rs)
        y = 100 * sum(x["best_pick"] == "yes" for x in rs) / len(rs)
        cf = sum(int(x["confidence"]) for x in rs) / len(rs)
        print(f"{d:16s} {len(rs):4d} {a:8.1f} {y:14.1f} {cf:10.2f}")
    print("  T1 is the attention check -- expect ~100%. T2 is the number the")
    print("  paper needs: models sit at 2-11% there.")

    t4 = [r for r in rows if r["direction"] == "T4_conflict" and r["force"]]
    if t4:
        print("\n--- T4 by event force: does the push/pull split hold up? ---")
        for f in ("pull", "push"):
            rs = [r for r in t4 if r["force"] == f]
            if not rs:
                continue
            a = 100 * sum(x["correct"] for x in rs) / len(rs)
            y = 100 * sum(x["best_pick"] == "yes" for x in rs) / len(rs)
            print(f"  {f:5s} n={len(rs):3d}  agrees with the key {a:5.1f}%   "
                  f"answer clearly best {y:5.1f}%")
        print("  The split is CONFIRMED if pull agreement is high and push")
        print("  agreement is low -- i.e. readers keep the stated preference")
        print("  when the event does not bear on it. If push agreement is also")
        print("  high, the `force` annotation is wrong and T4 can be pooled.")

    # ---- item-level usability, derived from the annotations themselves -----
    # A domain-level "constraint vs shading" split does not survive contact with
    # the data: exercise removes an option but the replacement is still not
    # clearly right, and spice_fwd and spice_rev fall on opposite sides. The
    # usable cut is per item, on the two judgements the annotators actually make.
    from collections import defaultdict as _dd
    cells = _dd(list)
    for r in rows:
        cells[(r["old_ok"], r["best_pick"])].append(r)
    print("\n--- item usability (old_ok x best_pick) ---")
    LABEL = {
        ("no", "yes"): "CLEAN        forced choice and appropriateness both valid",
        ("no", "no"): "ELIMINATION  appropriateness only; the replacement is not "
                      "clearly right",
        ("yes", "yes"): "SHADING      old still fine, new merely better",
        ("yes", "no"): "AMBIGUOUS    exclude",
    }
    for k in (("no", "yes"), ("no", "no"), ("yes", "yes"), ("yes", "no")):
        rs = cells.get(k, [])
        if not rs:
            continue
        a = 100 * sum(x["correct"] for x in rs) / len(rs)
        print(f"  old_ok={k[0]:7s} best={k[1]:7s} n={len(rs):4d}  "
              f"agreement {a:5.1f}%   {LABEL[k]}")

    # majority label per item, written out so the model runs can filter on it
    per_item_cells = _dd(list)
    for r in rows:
        per_item_cells[r["item"]].append((r["old_ok"], r["best_pick"]))
    labels = {}
    for iid, cs in per_item_cells.items():
        top = Counter(cs).most_common(1)[0][0]
        labels[iid] = dict(old_ok=top[0], best_pick=top[1],
                           usable_forced_choice=bool(top == ("no", "yes")),
                           usable_appropriateness=bool(top[0] == "no"),
                           n_ratings=len(cs),
                           unanimous=bool(len(set(cs)) == 1))
    out_p = os.path.join(dir_, "item_labels.json")
    with open(out_p, "w") as f:
        json.dump(labels, f, indent=2)
    nfc = sum(v["usable_forced_choice"] for v in labels.values())
    nap = sum(v["usable_appropriateness"] for v in labels.values())
    print(f"\n  wrote {out_p}")
    print(f"  {nfc}/{len(labels)} items usable for the forced choice, "
          f"{nap}/{len(labels)} for the appropriateness question")

    # items a single rater answered inconsistently, or raters disagreed on
    split = {k: v for k, v in per_item.items() if len(set(v)) > 1}
    if split:
        print(f"\n  {len(split)} items with conflicting ANSWERS across ratings:")
        for k, v in list(split.items())[:8]:
            print(f"    {k}: {sorted(set(v))}")
        print("  An item answered two different ways is not measuring anything")
        print("  stable. Drop it or rewrite the event.")

    multi = {k: v for k, v in per_item.items() if len(v) > 1}
    if multi:
        agree = sum(len(set(v)) == 1 for v in multi.values()) / len(multi)
        print(f"\ninter-annotator agreement on {len(multi)} doubly-rated items: "
              f"{100*agree:.1f}%")
    else:
        print("\n(no items rated by more than one annotator; assign overlap to "
              "get an agreement number)")

    print("\nDomains flagged above are candidates for exclusion. Decide and record "
          "the rule BEFORE looking at any further model results.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "ingest", "score"])
    ap.add_argument("--out", default="annotation_v2",
                    help="NOT 'annotation' -- that holds the v1 set")
    ap.add_argument("--dir", default="annotation_v2")
    ap.add_argument("--annotators", default="A,B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--items", default="data/items_v2.json")
    ap.add_argument("--turns", type=int, default=4,
                    help="must match the cell the models ran")
    ap.add_argument("--pos", type=float, default=0.6)
    a = ap.parse_args()
    if a.mode == "build":
        build(a.out, [x.strip() for x in a.annotators.split(",") if x.strip()],
              a.seed, a.items, a.turns, a.pos)
    elif a.mode == "ingest":
        ingest(a.dir)
    else:
        score(a.dir)
