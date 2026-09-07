#!/usr/bin/env python3
"""LLM judge over the reasoning traces. Asks what the trace itself concluded.

    export OPENAI_API_KEY=sk-...
    python judge_traces.py results/generate_thinkon__phase3_thinkon.csv --dry-run
    python judge_traces.py results/generate_thinkon__phase3_thinkon.csv
    python judge_traces.py ... --score        # re-report from the cache, no calls

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
Already settled by regex over the rehearsed <answer> tags inside the traces,
and validated on the conditions whose answer is known (T0 and T1 rehearse the
correct value in 100% of traces and answer correctly 100% of the time; T3
rehearses at 68.2% and answers at 68.2%):

    T2 traces rehearse the CORRECT value 10.8% of the time and the STALE value
    95.4%. The final answer differs from the last rehearsal on 2.3% of rows.

So the model is NOT deriving the new preference and then discarding it. It
commits early and deliberation does not revise. What regex cannot tell us is
WHERE the chain breaks, and that is this judge's job:

    does the trace work out what the event implies about the user's situation?
    if it does, does it connect that to the recommendation?
    if it declines to switch, on what grounds?

BLINDING
------------------------------------------
A judge that can see the answer key or the model's own answer will rationalise
whatever it is shown. So the judge gets:

  * the trace, with every <answer>...</answer> block and </think> marker
    scrubbed -- the model rehearses its answer inside the trace on 1099/1200
    rows, which would hand the verdict over directly
  * the two options, in an order randomised per item, never labelled
    old/new/correct
  * no setup text, no event text, no key, no final answer

The judge names an option by its literal word; the mapping to correct-vs-stale
happens here, after the call. T1 is included as calibration: a judge that
cannot see the conclusion in a T1 trace, where the event states the answer
outright, is not measuring anything and its T2 numbers should be discarded.

TRACE TRUNCATION
-------------------------------------------------------
For the phase-3 run the stored `reasoning` is the FIRST 3000 characters of a
median 6110-character trace, i.e. about half, head-first. The judge therefore
sees an opening, not a conclusion, and `cut_off` will be true on most rows.
That is a floor on engagement, not a measure of it. For the definitive number,
re-run generation with the trace kept whole:

    modal run modal_runner.py --experiment generate --thinking \\
        --raw-chars 24000 ... --tag phase3_full

"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

MODEL = "gpt-5-mini"
CACHE = "cache/judge_cache.jsonl"
os.makedirs("cache", exist_ok=True)   # not in the repo, so may not exist

ANSWER_TAG = re.compile(r"<answer>.*?</answer>", re.S | re.I)
ANSWER_OPEN = re.compile(r"<answer>.*", re.S | re.I)
THINK = re.compile(r"</?think>", re.I)

SYSTEM = (
    "You analyse a language model's private reasoning trace. You are told the "
    "two options it was choosing between, but NOT which is correct and NOT "
    "what it finally answered. Report only what the trace itself shows. Do not "
    "guess at a correct answer, and do not reason about the task yourself."
)

PROMPT = """A user told an assistant, in an earlier conversation, which option they preferred for their {noun}. Something later happened in their life. The assistant is now choosing between two options and this is its private reasoning.

The two options are: "{a}" and "{b}".

--- reasoning trace (may be cut off mid-sentence) ---
{trace}
--- end of trace ---

Answer these about the TRACE ONLY:

1. does it identify a specific event or change in the user's circumstances, beyond restating a stated preference?
2. does it work out a CONSEQUENCE of that event for the user -- something that is now harder, easier, required or ruled out?
3. does it connect that consequence to a recommendation?
4. which option does the trace favour? Answer with the option's exact word.
5. if it favours the option the user originally stated, what is its stated ground for staying?
6. is the trace cut off before it reaches a conclusion?

Read the WHOLE trace. In `spans_consequence`, quote verbatim every passage where the trace works out a consequence of the event -- as many as there are, of whatever length is needed to show the reasoning. Do NOT include passages that merely restate the user's stated preference. In `spans_conclusion`, quote verbatim every passage where it settles on or reconsiders an option. Empty lists if there are none."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["identifies_event", "derives_consequence", "links_to_choice",
                 "favours", "ground_for_staying", "cut_off", "confidence",
                 "spans_consequence", "spans_conclusion"],
    "properties": {
        "identifies_event": {"type": "boolean"},
        "derives_consequence": {"type": "boolean"},
        "links_to_choice": {"type": "boolean"},
        "favours": {"type": "string"},
        "ground_for_staying": {
            "type": "string",
            "enum": ["not_applicable", "stated_preference_is_authoritative",
                     "event_judged_irrelevant", "event_not_noticed",
                     "other_option_judged_worse", "unclear"]},
        "cut_off": {"type": "boolean"},
        # strict structured-output mode supports only a keyword subset:
        # minimum/maximum are rejected with a 400, so the range is an enum.
        "confidence": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "spans_consequence": {
            "type": "array", "items": {"type": "string"},
            "description": "every verbatim passage where the trace derives a "
                           "consequence of the event. No length limit. Never "
                           "a restatement of the user's stated preference."},
        "spans_conclusion": {
            "type": "array", "items": {"type": "string"},
            "description": "every verbatim passage where the trace settles on "
                           "or reconsiders an option. No length limit."},
    },
}


def scrub(trace):
    """Remove anything that hands the model's own answer to the judge."""
    t = ANSWER_TAG.sub(" [redacted] ", str(trace))
    t = ANSWER_OPEN.sub(" [redacted]", t)      # unclosed tag at a clip boundary
    t = THINK.sub("", t)
    return t.strip()


def build(row, noun, seed):
    """-> (prompt, schema, options) with the presentation order randomised.

    The options are named by their literal word, not as A/B. The trace restates
    the model's OWN "A. x / B. y" listing, which is a different assignment from
    any we would impose here, so an A/B answer would be ambiguous about whose
    labelling it meant. A schema enum of the two words removes that entirely.
    """
    rng = random.Random(f"{row['id']}-{seed}")
    opts = [row["opt_a"], row["opt_b"]]
    rng.shuffle(opts)
    p = PROMPT.format(noun=noun, a=opts[0], b=opts[1],
                      trace=scrub(row["reasoning"]))
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["favours"]["enum"] = opts + ["neither", "unclear"]
    return p, schema, opts


def _cache_load(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                out[r["key"]] = r["verdict"]
            except Exception:
                continue
    return out


def _key(model, prompt):
    return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()[:32]


def call_openai(client, model, prompt, effort, schema, max_out):
    kw = dict(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "trace_verdict", "strict": True, "schema": schema}},
    )
    # GPT-5 reasoning models take max_completion_tokens and reasoning_effort,
    # and reject temperature. Older chat models take max_tokens. Try the modern
    # shape, fall back once, and surface anything else.
    for attempt in ({"max_completion_tokens": max_out,
                     "reasoning_effort": effort},
                    {"max_completion_tokens": max_out},
                    {"max_tokens": max_out}):
        try:
            r = client.chat.completions.create(**kw, **attempt)
            return json.loads(r.choices[0].message.content)
        except TypeError:
            continue
        except Exception as e:
            msg = str(e)
            if any(w in msg for w in ("unsupported", "Unrecognized",
                                      "unexpected keyword", "not supported")):
                continue
            raise
    raise RuntimeError("no accepted parameter shape for this model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--conditions", default="T1_explicit,T2_implicit,"
                                            "T3_counterfact,T4_conflict",
                    help="T1 and T3 are calibration; keep them")
    ap.add_argument("--per-condition", type=int, default=0,
                    help="0 = judge EVERY trace in the condition (default). "
                         "Set a number only to deliberately subsample.")
    ap.add_argument("--max-out", type=int, default=32000,
                    help="cap on the judge's own output tokens. A reasoning "
                         "model spends these on thinking too, so a low value "
                         "truncates its reply into malformed JSON. This is the "
                         "one limit the API requires; raise it if you see "
                         "length errors.")
    ap.add_argument("--effort", default="low",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="print two prompts and the blinding check, call nothing")
    ap.add_argument("--score", action="store_true",
                    help="report from the cache only, make no calls")
    a = ap.parse_args()

    import pandas as pd
    d = pd.read_csv(a.csv)
    items = {i["id"]: i for i in json.load(open("data/items_v2.json"))}
    want = [c.strip() for c in a.conditions.split(",") if c.strip()]
    d = d[d.condition.isin(want) & d.parsed & d.reasoning.notna()].copy()
    if not len(d):
        sys.exit("no rows with a stored reasoning trace match those conditions")

    rng = random.Random(a.seed)
    picked = []
    for c in want:
        g = d[d.condition == c].to_dict("records")
        rng.shuffle(g)
        picked.extend(g if a.per_condition <= 0 else g[:a.per_condition])
    print(f"{a.csv}: judging {len(picked)} traces with {a.model} "
          f"(effort={a.effort})")
    print("  per condition:", dict(Counter(r["condition"] for r in picked)))

    jobs = []
    for r in picked:
        noun = items[r["id"]]["noun"]
        prompt, schema, opts = build(r, noun, a.seed)
        jobs.append((r, prompt, schema, opts))

    if a.dry_run:
        print("\n=== BLINDING CHECK over all selected traces ===")
        bad = []
        for r, prompt, schema, opts in jobs:
            body = prompt.split("--- reasoning trace")[1]
            if re.search(r"<answer>", body, re.I):
                bad.append((r["id"], "answer tag survived scrubbing"))
            if str(r["picked"]) and re.search(
                    rf"\bfinal answer\b", body, re.I):
                bad.append((r["id"], "phrase 'final answer' in trace"))
        print(f"  answer tags leaked: "
              f"{sum(1 for _, w in bad if 'tag' in w)} of {len(jobs)}")
        print(f"  correct answer named in the prompt outside the option list: "
              "n/a -- the key is never sent")
        pos = Counter(o[0] == r["correct"] for r, _, _, o in jobs)
        print(f"  correct option listed first on {pos[True]} and second on "
              f"{pos[False]} items (order randomised per item)")
        for r, prompt, schema, opts in jobs[:2]:
            print("\n" + "=" * 72)
            print(f"[{r['condition']}]  enum={schema['properties']['favours']['enum']}  "
                  f"(key={r['correct']}, model said={r['picked']} -- NOT sent)")
            print("=" * 72)
            print(prompt[:2200])
        return

    cache = _cache_load(a.cache)
    todo = [j for j in jobs if _key(a.model, j[1]) not in cache]
    print(f"  {len(jobs) - len(todo)} cached, {len(todo)} to fetch")

    if todo and not a.score:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("pip install openai")
        client = OpenAI()
        from concurrent.futures import ThreadPoolExecutor
        fh = open(a.cache, "a")
        done = [0]

        def work(job):
            r, p, sch, o = job
            try:
                v = call_openai(client, a.model, p, a.effort, sch, a.max_out)
            except Exception as e:
                return r, p, {"error": str(e)}
            return r, p, v

        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            for r, p, v in ex.map(work, todo):
                done[0] += 1
                if "error" not in v:
                    fh.write(json.dumps({"key": _key(a.model, p),
                                         "verdict": v}) + "\n")
                    fh.flush()
                    cache[_key(a.model, p)] = v
                elif done[0] <= 3:
                    print(f"  ! {r['id']}: {v['error']}")
                if done[0] % 25 == 0:
                    print(f"  {done[0]}/{len(todo)}", flush=True)
        fh.close()

    rows = []
    for r, p, sch, o in jobs:
        v = cache.get(_key(a.model, p))
        if not v:
            continue
        fav = str(v["favours"])
        fav_label = fav if fav in (r["opt_a"], r["opt_b"]) else None
        old = items[r["id"]]["old"]
        rows.append(dict(
            id=r["id"], condition=r["condition"],
            identifies_event=v["identifies_event"],
            derives_consequence=v["derives_consequence"],
            links_to_choice=v["links_to_choice"],
            cut_off=v["cut_off"], confidence=v["confidence"],
            ground=v["ground_for_staying"],
            n_spans_consequence=len(v.get("spans_consequence") or []),
            n_spans_conclusion=len(v.get("spans_conclusion") or []),
            spans_consequence=" || ".join(v.get("spans_consequence") or []),
            spans_conclusion=" || ".join(v.get("spans_conclusion") or []),
            trace_favours=fav_label if fav_label else fav,
            trace_favours_correct=(fav_label == r["correct"]),
            trace_favours_stale=(fav_label == old),
            model_answered_correct=bool(r["is_correct"]),
        ))
    if not rows:
        sys.exit("nothing in the cache for these prompts yet")
    j = pd.DataFrame(rows)
    out = a.out or a.csv.replace(".csv", "_judged.csv")
    j.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(j)} verdicts)\n")

    print("=== CALIBRATION: does the judge read a trace correctly at all? ===")
    cal = j[j.condition.isin(["T1_explicit", "T0_no_evidence"])]
    if not len(cal):
        print("  !! NO CALIBRATION CONDITION IN THIS RUN.")
        print("  T1 states the answer outright, so a judge that cannot see the")
        print("  conclusion there is not reading conclusions at all. Without it")
        print("  the numbers below are UNVALIDATED and must not be quoted.")
        print("  Re-run including T1: --conditions T1_explicit,T2_implicit\n")
    else:
        bad = False
        for c in ("T1_explicit", "T0_no_evidence"):
            g = cal[cal.condition == c]
            if not len(g):
                continue
            jc = 100 * g.trace_favours_correct.mean()
            mc = 100 * g.model_answered_correct.mean()
            flag = ""
            if mc - jc > 20:
                flag = "   <-- judge is missing conclusions the model reached"
                bad = True
            print(f"  {c}: judge reads the trace as favouring the key on "
                  f"{jc:.1f}%; the model answered correctly on {mc:.1f}%{flag}")
        if bad:
            print("  >>> the judge under-reads conclusions on a condition where")
            print("      the answer is stated outright. Do not quote the")
            print("      engagement numbers below.\n")
        else:
            print("  Judge and model agree on the calibration condition, so the "
                  "judge is reading conclusions.\n")

    print("=== WHERE THE CHAIN BREAKS ===")
    print(f"{'condition':16s} {'n':>4s} {'ident. event':>13s} {'derives':>9s} "
          f"{'links':>7s} {'cut off':>8s} {'favours key':>12s} {'model right':>12s}")
    for c in want:
        g = j[j.condition == c]
        if not len(g):
            continue
        print(f"{c:16s} {len(g):4d} {100*g.identifies_event.mean():12.1f}% "
              f"{100*g.derives_consequence.mean():8.1f}% "
              f"{100*g.links_to_choice.mean():6.1f}% "
              f"{100*g.cut_off.mean():7.1f}% "
              f"{100*g.trace_favours_correct.mean():11.1f}% "
              f"{100*g.model_answered_correct.mean():11.1f}%")

    print("\n=== when the trace keeps the stated preference, why? ===")
    st = j[j.trace_favours_stale]
    for c in want:
        g = st[st.condition == c]
        if not len(g):
            continue
        print(f"  {c} (n={len(g)}): "
              f"{dict(Counter(g.ground).most_common())}")

    print("\n=== does the trace's conclusion match what the model answered? ===")
    for c in want:
        g = j[j.condition == c]
        if not len(g):
            continue
        agree = (g.trace_favours_correct == g.model_answered_correct).mean()
        print(f"  {c}: {100*agree:5.1f}% agreement "
              f"(disagreement is where deliberation and commitment diverge)")

    print("\n=== AUDIT: is `derives_consequence` justified by its quote? ===")
    s = j[j.derives_consequence]
    if not len(s):
        print("  no trace in this run was flagged as deriving a consequence.")
    print(f"  (conditions in this run: "
          f"{dict(j.condition.value_counts())})")
    setupish = s.spans_consequence.fillna("").str.contains(
        "what works for me", case=False)
    empty = s.n_spans_consequence.fillna(0).eq(0)
    print(f"  {len(s)} traces flagged as deriving a consequence")
    # T0 carries NO event at all, so any flag there is a false positive and
    # sets the floor the other conditions must be read against.
    t0 = j[j.condition == "T0_no_evidence"]
    if len(t0):
        print(f"\n  FALSE-POSITIVE FLOOR from T0, which contains no event:")
        print(f"    identifies_event   {100*t0.identifies_event.mean():5.1f}%")
        print(f"    derives_consequence{100*t0.derives_consequence.mean():5.1f}%")
        print(f"    links_to_choice    {100*t0.links_to_choice.mean():5.1f}%")
        print("    Subtract these before reading any other condition. The")
        print("    filler turns are themselves small life events, so a trace")
        print("    discussing one can look like event engagement.")
    print(f"    quote is the user's stated preference: {int(setupish.sum())}")
    print(f"    quote is empty:                        {int(empty.sum())}")
    if len(s) and (setupish.sum() + empty.sum()) / len(s) > 0.3:
        print("  >>> most flags are unsupported by their own quote. Treat")
        print("      derives_consequence as unreliable in this run.")
    for _, r in s.head(6).iterrows():
        mark = "CORRECT" if r.model_answered_correct else "STALE  "
        print(f"\n  [{mark}] favours={r.trace_favours!r}  "
              f"({r.n_spans_consequence} consequence span(s), "
              f"{r.n_spans_conclusion} conclusion span(s))")
        print(f"      consequence: {r.spans_consequence}")
        print(f"      conclusion : {r.spans_conclusion}")


if __name__ == "__main__":
    main()
