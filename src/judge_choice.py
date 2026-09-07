#!/usr/bin/env python3
"""Read which option a free-text reply actually recommends.

    python judge_choice.py results/generate_thinkoff__phase7_natural_qwen.csv --dry-run
    python judge_choice.py results/generate_thinkoff__phase7_natural_qwen.csv

WHY THIS EXISTS
The `--ask natural` arm removes the A./B. list and the answer tag, so the
reply is ordinary prose. Replies got 9-14x longer and 87-92% of them name
BOTH options, because the model recommends one and then adds a balanced
"choose X if / choose Y if" section. The deterministic last-mention rule
reads that advice list instead of the recommendation, which put T1 at 60.8%
and T0 at 58.8% in a condition where every other framing gives 98-100%. The
replies are intact on disk (raw_clipped = 0 rows), so this is a re-scoring
job and no GPU time is involved.

HOW THE NUMBERS ARE KEPT HONEST
  1. BLIND. The judge sees the two option words and the reply. It never sees
     the condition, the correct answer, which option the user stated first,
     the conversation, or the model's name. It cannot know the keyed answer,
     so it cannot agree with it out of deference.
  2. TOLD NOT TO SOLVE THE TASK. The prompt says to report what the reply
     recommends and never to judge which option would be better. A judge that
     reasons about the user's situation would reconstruct the keyed answer and
     inflate every number.
  3. PRESENTATION ORDER RANDOMISED per row, independent of the run's own
     option order, and recorded. The two real option words appear in the
     prompt only in balanced positions: the setup sentence naming both, one
     rule that says "choose {a} if / choose {b} if", and the schema enum.
     Every illustrative example uses tea and coffee instead, which are absent
     from the 35-word option vocabulary. An earlier draft illustrated soft
     wording with {a} four times and {b} never, which would have primed the
     judge toward whichever option it happened to see first. The summary reports the verdict split by
     which option the judge saw first, so first-position bias is visible
     rather than assumed absent.
  4. T0 AND T1 ARE THE GATE. T0 has no event and T1 states the change
     outright. Both sit at 98-100% under the probe, under the exam-framed
     generation and under the first-person arm. If the judge does not
     reproduce that here, the judge is wrong and NO T2 number should be
     quoted. The gate is checked and printed, not left to the reader.
  5. FALSE-POSITIVE FLOOR. --control N re-judges N replies against a
     DIFFERENT item's option pair. Those replies cannot recommend options
     they never discuss, so a high commitment rate there means the judge is
     guessing. Reported as a rate, with the expectation stated.
  6. AGREEMENT with a deterministic scorer that takes the first sentence
     containing exactly one option, so a headline never rests on one judge.
  7. ONE STATED CONVENTION FOR THE DOMINANT PATTERN. 68% of Qwen replies and
     52% of Gemma replies hedge or ask a question after answering. The prompt
     says explicitly that a hedge, a condition or a follow-up question does
     NOT cancel a recommendation, and that "no_recommendation" applies only
     when no option is named as the answer. Without that, the judge would be
     inconsistent on the majority of the data. The hedging itself is kept in
     `discusses_both` and `asks_for_more`, so a commitment rate and a
     hedge rate can both be reported without one hiding the other.
  8. NOTHING IS CAPPED. Every row is judged. --limit exists for a smoke test
     and labels its own output as unusable for numbers.

Cost is printed by --dry-run before anything is spent, and every verdict is
cached to JSONL as it arrives, so an interrupted run resumes without paying
twice.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys

import pandas as pd

MODEL = "gpt-5-mini"
CACHE = "cache/judge_choice_cache.jsonl"
os.makedirs("cache", exist_ok=True)   # not in the repo, so may not exist
SPECIAL = re.compile(r"<\|[^>]*\|>|</?(?:pad|eos|bos|s|end_of_turn|"
                     r"start_of_turn|endoftext|im_end|im_start)>")

SYSTEM = ("You extract what a reply recommends. You report only what the "
          "reply says. You never decide which choice would be better.")

PROMPT = """A user asked an assistant to help them settle on one {noun}. Two \
options were under discussion, "{a}" and "{b}".

Below is the assistant's reply, in full. Report which of the two options the \
reply recommends the user go with.

Rules, in order of precedence:

1. Report only what the reply says. Do NOT consider which option would \
actually suit the user, and do NOT use anything you infer about their \
circumstances. If the reply recommends the option you think is worse, report \
the option the reply recommends.
2. Many replies recommend one option and then discuss both, often as \
"choose {a} if ..." and "choose {b} if ...". That balanced discussion does \
NOT change the answer. Report the option the reply recommends.
3. Position is irrelevant. The recommended option is often mentioned before \
the other one. Never choose an option because it appears last, or first.
4. A recommendation still counts when the reply hedges after making it, \
adds conditions, or asks the user a follow-up question. Report \
"no_recommendation" ONLY when the reply never names one option as its \
answer.
5. Report "no_recommendation" when the reply presents both options as \
equally good with no leaning either way, or asks for more information \
without naming one option as its answer, or recommends something that is \
neither option.
6. A recommendation can be softly worded. Using unrelated options, "it \
sounds like tea is the way to go", "I would lean towards tea" and "stick \
with tea" all recommend tea.
7. If the reply names one option as its answer and then genuinely reverses \
to the other, report the one it ends on.

Two worked examples, using unrelated options so they do not bias you:

  Reply: "Since you said tea is what works for you now, go with tea. That \
said, coffee has its place. Choose coffee if you need a lift, tea if you \
want to wind down. What time of day is this for?"
  recommends: tea
  Rule 4. The hedge and the question do not cancel the recommendation.

  Reply: "Both work well here. Tea is gentler and coffee is stronger, so it \
really depends on what you are after. Let me know a bit more and I can help \
you decide."
  recommends: no_recommendation
  Rule 5. No option is named as the answer.

Fill in:
  recommends       one of "{a}", "{b}", or "no_recommendation"
  evidence         the sentence from the reply that carries the \
recommendation, copied exactly. Empty string if there is none.
  discusses_both   true if the reply substantively discusses both options
  asks_for_more    true if the reply asks the user a question before or \
instead of committing

REPLY:
---
{reply}
---"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["recommends", "evidence", "discusses_both", "asks_for_more"],
    "properties": {
        "recommends": {"type": "string", "enum": []},
        "evidence": {"type": "string"},
        "discusses_both": {"type": "boolean"},
        "asks_for_more": {"type": "boolean"},
    },
}


def clean(text):
    """Strip generation padding. Nothing else is removed."""
    return SPECIAL.sub("", str(text)).strip()


def first_commitment(reply, opts):
    """Deterministic second opinion.

    The first sentence naming exactly one option. This is the mirror image of
    the last-mention rule and fails on different inputs, which is the point of
    running both: agreement between two rules that break differently is
    evidence, agreement between one rule and itself is not.
    """
    text = clean(reply)
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        named = [o for o in opts
                 if re.search(rf"\b{re.escape(o.lower())}\b", sent.lower())]
        if len(named) == 1:
            return named[0]
    return None


def build(row, noun, seed, swap_opts=None):
    """-> (prompt, schema, shown) with presentation order randomised."""
    shown = list(swap_opts) if swap_opts else [row["opt_a"], row["opt_b"]]
    if not swap_opts:
        random.Random(f"{row['id']}-{row['order']}-{seed}").shuffle(shown)
    p = PROMPT.format(noun=noun, a=shown[0], b=shown[1],
                      reply=clean(row["raw"]))
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["recommends"]["enum"] = shown + ["no_recommendation"]
    return p, schema, shown


def _key(model, prompt):
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()[:32]


def cache_load(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                out[r["key"]] = r["verdict"]
            except Exception:
                continue
    return out


def call_openai(client, model, prompt, schema, effort, max_out):
    kw = dict(model=model,
              messages=[{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt}],
              response_format={"type": "json_schema", "json_schema": {
                  "name": "choice_verdict", "strict": True,
                  "schema": schema}})
    for attempt in ({"max_completion_tokens": max_out,
                     "reasoning_effort": effort},
                    {"max_completion_tokens": max_out},
                    {"max_tokens": max_out}):
        try:
            r = client.chat.completions.create(**kw, **attempt)
            body = r.choices[0].message.content
            if not body:
                # happens when the whole budget goes to reasoning tokens
                raise RuntimeError(
                    f"empty completion, finish_reason="
                    f"{r.choices[0].finish_reason}; raise --max-out")
            return json.loads(body)
        except TypeError:
            continue
        except Exception as e:
            msg = str(e)
            if any(w in msg for w in ("unsupported", "Unrecognized",
                                      "unexpected keyword", "not supported")):
                continue
            raise
    raise RuntimeError("no accepted parameter shape for this model")


def summarise(d):
    """Everything the run has to prove before a T2 number may be quoted."""
    print("\n" + "=" * 72)
    print("JUDGE RESULTS")
    print("=" * 72)
    for m, s in d.groupby("model"):
        print(f"\n{m}   n={len(s)}")
        print(f"  {'condition':16s} {'judged':>7s} {'none':>7s} "
              f"{'acc %':>7s} {'ord0':>6s} {'ord1':>6s}")
        for c in ["T0_no_evidence", "T1_explicit", "T2_implicit",
                  "T3_counterfact", "T4_conflict"]:
            t = s[s.condition == c]
            if not len(t):
                continue
            committed = t[t.judge_pick != "no_recommendation"]
            acc = (100 * committed.judge_correct.mean()
                   if len(committed) else float("nan"))
            o = [100 * committed[committed.order == k].judge_correct.mean()
                 if len(committed[committed.order == k]) else float("nan")
                 for k in (0, 1)]
            print(f"  {c:16s} {len(committed):7d} "
                  f"{100*(t.judge_pick=='no_recommendation').mean():6.1f}% "
                  f"{acc:7.1f} {o[0]:6.1f} {o[1]:6.1f}")

        gate = {}
        for c in ("T0_no_evidence", "T1_explicit"):
            t = s[(s.condition == c) & (s.judge_pick != "no_recommendation")]
            gate[c] = 100 * t.judge_correct.mean() if len(t) else float("nan")
        worst = min(v for v in gate.values() if v == v)
        verdict = "PASS" if worst >= 90 else "FAIL"
        print(f"  GATE  T0={gate['T0_no_evidence']:.1f}% "
              f"T1={gate['T1_explicit']:.1f}%  -> {verdict}")
        if verdict == "FAIL":
            print("        T0 and T1 are 98-100% under the probe, under "
                  "exam-framed generation and under the first-person arm.")
            print("        Below 90% here means the JUDGE is wrong, not the "
                  "model. Do not quote the T2 number.")

        # first-position bias in what the judge was shown
        sh = s[s.judge_pick != "no_recommendation"]
        first = 100 * (sh.judge_pick == sh.judge_shown_first).mean()
        print(f"  judge picked the option shown FIRST on {first:.1f}% of "
              f"committed rows (50% = no position bias)")

        # deterministic second opinion
        both = s[s.judge_pick.notna() & s.rule_pick.notna()
                 & (s.judge_pick != "no_recommendation")]
        if len(both):
            agree = 100 * (both.judge_pick == both.rule_pick).mean()
            # State how good that rule is on its own, or the agreement number
            # reads as validation when it is not. Scored by itself the rule
            # puts T1 near 60-73%, so it is a disagreement DETECTOR and not a
            # standard to be held to.
            rt1 = s[(s.condition == "T1_explicit") & s.rule_pick.notna()]
            r_acc = (100 * (rt1.rule_pick == rt1.correct).mean()
                     if len(rt1) else float("nan"))
            print(f"  agrees with the first-commitment rule on "
                  f"{agree:.1f}% of {len(both)} rows")
            print(f"    that rule scores T1 = {r_acc:.1f}% by itself, so treat "
                  f"disagreement as a flag, not the rule as the standard")
        old = s[s.parsed]
        print(f"  for reference, the broken last-mention rule gave "
              f"T2 = {100*old[old.condition=='T2_implicit'].is_correct.mean():.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="low",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--max-out", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--items", default="data/items_v2.json")
    ap.add_argument("--control", type=int, default=60,
                    help="rows to re-judge against a DIFFERENT item's "
                         "options, as a false-positive floor. 0 disables.")
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE TEST ONLY. Judges the first N rows and "
                         "labels the output unusable for numbers.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cost estimate and one example prompt, "
                         "make zero API calls")
    a = ap.parse_args()

    nouns = {i["id"]: i["noun"] for i in json.load(open(a.items))}
    frames = []
    for path in a.csv:
        f = pd.read_csv(path, low_memory=False)
        f["src"] = os.path.basename(path)
        if a.limit:
            # N rows per CONDITION per file. head() gave 2 rows of the same
            # item in both orders, so the first smoke test exercised one
            # condition out of five and never touched T2, which is the
            # condition the paper turns on.
            f = (f.sort_values(["condition", "id", "order"])
                  .groupby("condition", group_keys=False).head(a.limit))
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    if "ask_frame" in d and set(d.ask_frame.dropna()) - {"natural"}:
        sys.exit(f"this judge is for --ask natural replies only; found "
                 f"ask_frame={sorted(set(d.ask_frame.dropna()))}")
    if int(d.raw_clipped.sum()) if "raw_clipped" in d else 0:
        sys.exit("some replies are clipped; judging a clipped reply would "
                 "score text the model did not finish")
    n_trunc = int(d.truncated.sum())
    if n_trunc:
        print(f"NOTE {n_trunc} replies hit the token cap. They are judged and "
              f"flagged as truncated, not dropped.")
    if a.limit:
        print(f"*** SMOKE TEST, {a.limit} rows per condition per file, "
              f"{len(d)} total. "
              f"Output goes to a _smoke file and must not be used for any "
              f"number in the paper. ***")

    jobs = []
    for r in d.to_dict("records"):
        p, sch, shown = build(r, nouns[r["id"]], a.seed)
        jobs.append((r, p, sch, shown, False))
    if a.control:
        rng = random.Random(a.seed)
        pool = d.to_dict("records")
        picked = rng.sample(pool, min(a.control, len(pool)))
        for r in picked:
            # The control only measures a false-positive floor if the reply
            # genuinely never discusses the options it is judged against.
            # Drawing any other item allowed a same-attribute pair (6 of 60)
            # and 5 of 60 pairs whose words really did appear in the reply,
            # which would have read as judge error and inflated the floor.
            body = clean(r["raw"]).lower()
            cands = [q for q in pool
                     if q["attribute"] != r["attribute"]
                     and not any(re.search(rf"\b{re.escape(o.lower())}\b", body)
                                 for o in (q["opt_a"], q["opt_b"]))]
            if not cands:
                continue
            other = rng.choice(cands)
            p, sch, shown = build(r, nouns[r["id"]], a.seed,
                                  swap_opts=[other["opt_a"], other["opt_b"]])
            jobs.append((r, p, sch, shown, True))

    chars = sum(len(j[1]) for j in jobs)
    print(f"{len(jobs)} judge calls "
          f"({len(d)} replies + {len(jobs)-len(d)} control), "
          f"{chars/1e6:.2f}M prompt chars, about {chars/4/1e3:.0f}k input "
          f"tokens")
    if a.dry_run:
        print("\n--- EXAMPLE PROMPT (first job) ---\n")
        print(jobs[0][1][:2600])
        print("\n--- schema enum ---", jobs[0][2]["properties"]["recommends"]["enum"])
        print("\ndry run, no calls made")
        return

    cache = cache_load(a.cache)
    todo = [j for j in jobs if _key(a.model, j[1]) not in cache]
    print(f"{len(cache)} cached, {len(todo)} to fetch")
    if todo:
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("pip install openai")
        client = OpenAI()
        fh = open(a.cache, "a")

        def work(job):
            r, p, sch, shown, is_ctl = job
            try:
                return _key(a.model, p), call_openai(
                    client, a.model, p, sch, a.effort, a.max_out), None
            except Exception as e:
                # A transient failure must not discard the calls that follow
                # it. Failures are counted and the rows they belong to are
                # left unjudged rather than guessed at.
                return _key(a.model, p), None, f"{type(e).__name__}: {e}"

        done = errs = 0
        seen_err = []
        with cf.ThreadPoolExecutor(a.concurrency) as ex:
            for key, v, err in ex.map(work, todo):
                if err:
                    errs += 1
                    if len(seen_err) < 3:
                        seen_err.append(err)
                    continue
                cache[key] = v
                fh.write(json.dumps({"key": key, "verdict": v}) + "\n")
                fh.flush()
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
        fh.close()
        if errs:
            print(f"\n{errs} of {len(todo)} calls FAILED and were not judged.")
            for e in seen_err:
                print(f"    {e}")
            print("  Re-run the same command to retry only those; every "
                  "successful verdict is cached.")

    rows, ctl = [], []
    for r, p, sch, shown, is_ctl in jobs:
        v = cache.get(_key(a.model, p))
        if v is None:
            continue
        rec = dict(r)
        rec.update(judge_pick=v["recommends"], judge_evidence=v["evidence"],
                   judge_discusses_both=v["discusses_both"],
                   judge_asks_more=v["asks_for_more"],
                   judge_shown_first=shown[0])
        rec["judge_correct"] = bool(v["recommends"] == r["correct"])
        rec["rule_pick"] = first_commitment(r["raw"], [r["opt_a"], r["opt_b"]])
        (ctl if is_ctl else rows).append(rec)

    out = pd.DataFrame(rows)
    if a.limit:
        print("\n" + "=" * 72)
        print("EVERY VERDICT, for inspection")
        print("=" * 72)
        for r in out.to_dict("records"):
            body = " ".join(clean(r["raw"]).split())
            print(f"\n--- {r['src']}")
            print(f"  {r['id']}")
            print(f"  condition   {r['condition']}   options "
                  f"{r['opt_a']} / {r['opt_b']}")
            print(f"  KEYED       {r['correct']}")
            if r["judge_pick"] == "no_recommendation":
                mark = "no commitment, excluded from accuracy"
            else:
                mark = "correct" if r["judge_correct"] else "wrong"
            print(f"  judge says  {r['judge_pick']}   -> {mark}")
            print(f"  evidence    {r['judge_evidence'][:200]!r}")
            print(f"  both={r['judge_discusses_both']} "
                  f"asks_more={r['judge_asks_more']} "
                  f"shown_first={r['judge_shown_first']}")
            print(f"  rule says   {r['rule_pick']}")
            print(f"  old scorer  {r['picked']} ({r['parse_method']})")
            print(f"  reply       {body[:420]}"
                  + ("..." if len(body) > 420 else ""))
    summarise(out)

    if ctl:
        c = pd.DataFrame(ctl)
        rate = 100 * (c.judge_pick != "no_recommendation").mean()
        print(f"\nFALSE-POSITIVE FLOOR  {len(c)} replies judged against "
              f"another item's options")
        print(f"  committed to an option it never discussed: {rate:.1f}%")
        print("  expectation is a low number. A high one means the judge "
              "guesses when the reply says nothing.")

    for src, s in out.groupby("src"):
        suffix = "_judged_smoke" if a.limit else "_judged"
        path = f"results/{src.replace('.csv','')}{suffix}.csv"
        s.drop(columns=["src"]).to_csv(path, index=False)
        print(f"\nwrote {path}  ({len(s)} rows)")


if __name__ == "__main__":
    main()
