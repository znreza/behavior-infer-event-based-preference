#!/usr/bin/env python3
"""
Does HorizonBench actually contain the event as a conversation?

    python check_hb_events.py --n 60

WHY
---
Mining produced items whose event_name and event text are unrelated:

    event_name : Severe Vision Impairment from Lab Accident
    event text : "i just read this neuroscience article claiming free will..."

Date matching is landing on the wrong episode. Before trying to fix the matcher,
settle the prior question: is there an episode ANYWHERE in the conversation that
mentions the event? If not, no matcher will help, mined items cannot test
event-to-preference inference, and the same doubt applies to any earlier
experiment that located the event by date.

WHAT IT DOES
------------
For each evolved row, take the event_name from evolution_history, and score
EVERY episode in the conversation by content-word overlap with it. Report:

  best-overlap score distribution   how well the best episode matches
  same-day vs best-overlap agree    does date matching find that episode
  a few worked examples             so you can read them yourself

Interpretation:
  high overlap for most rows, and same-day usually equals best
      -> the episode exists; the matcher just needs to use overlap, not date
  high overlap, but same-day rarely equals best
      -> the episode exists but is dated differently; switch to overlap matching
  low overlap for most rows
      -> the event is graph-only. Mining cannot give you inference items.
"""

import argparse, json, re
from collections import Counter

DATASET = "stellalisy/HorizonBench"
EP_RE = re.compile(r"^Date:\s*(\S+)\s*\nScenario:\s*(.+)$", re.M)

STOP = set("""the a an and or of to in is are was were i you it that this for on with my me we they
he she but so if as at be been have has had do does did not no yes about from into over under
their there here what when where which who whom will would can could should than then them our
your his her its been being just really very much more most some any all one two""".split())


def words(s):
    return {w for w in re.findall(r"[a-z]{4,}", str(s).lower()) if w not in STOP}


def split_episodes(conv):
    hits = list(EP_RE.finditer(conv))
    out = []
    for k, h in enumerate(hits):
        start = h.start()
        end = hits[k + 1].start() if k + 1 < len(hits) else len(conv)
        out.append(dict(date=h.group(1), scenario=h.group(2).strip(),
                        text=conv[start:end], idx=k))
    return out


def main(n_rows, seed, show):
    from datasets import load_dataset
    import random
    ds = load_dataset(DATASET)
    D = ds["test" if "test" in ds else list(ds.keys())[0]]

    rng = random.Random(seed)
    idx = [i for i, h in enumerate(D["has_evolved"]) if h]
    rng.shuffle(idx)

    scores, agree, checked = [], [], 0
    examples = []
    for i in idx:
        if checked >= n_rows:
            break
        r = D[i]
        try:
            pe = json.loads(r["preference_evolution"])
        except Exception:
            continue
        hist = pe.get("evolution_history") or []
        if not hist:
            continue
        h = hist[0]
        ev_name, ev_date = h.get("event_name", ""), str(h.get("date", ""))[:10]
        q = words(ev_name)
        if not q:
            continue

        eps = split_episodes(r["conversation"])
        if len(eps) < 3:
            continue

        # score every episode: overlap of event_name content words with the
        # episode's scenario line AND its body
        best, best_s = None, -1.0
        for e in eps:
            hay = words(e["scenario"]) | words(e["text"][:1500])
            s = len(q & hay) / len(q)
            if s > best_s:
                best, best_s = e, s

        same_day = [e for e in eps if str(e["date"])[:10] == ev_date]
        sd_is_best = bool(same_day) and any(e["idx"] == best["idx"] for e in same_day)

        scores.append(best_s)
        agree.append(sd_is_best)
        checked += 1
        if len(examples) < show:
            examples.append((ev_name, ev_date, best_s, best, same_day[0] if same_day else None))

    import statistics as st
    print(f"\nchecked {checked} evolved rows")
    print(f"best-episode overlap with event_name:")
    print(f"   median {st.median(scores):.2f}   mean {sum(scores)/len(scores):.2f}")
    buckets = Counter("0.0-0.2" if s < .2 else "0.2-0.4" if s < .4 else
                      "0.4-0.6" if s < .6 else "0.6-0.8" if s < .8 else "0.8-1.0"
                      for s in scores)
    for b in ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"):
        n = buckets.get(b, 0)
        print(f"   {b}  {n:4d}  {'#' * int(40*n/max(checked,1))}")
    print(f"\nsame-day episode IS the best-overlap episode: "
          f"{100*sum(agree)/len(agree):.1f}% of rows")

    print("\n" + "=" * 72)
    for ev_name, ev_date, s, best, sd in examples:
        print(f"\nEVENT   : {ev_name}   [{ev_date}]")
        print(f"BEST ep : overlap {s:.2f}  date {best['date'][:10]}")
        print(f"          scenario: {best['scenario'][:110]}")
        if sd is not None and sd['idx'] != best['idx']:
            print(f"SAME-DAY: date {sd['date'][:10]}  (DIFFERENT episode)")
            print(f"          scenario: {sd['scenario'][:110]}")
        elif sd is None:
            print("SAME-DAY: no episode on that date")
    print("\n" + "=" * 72)

    med = st.median(scores)
    if med >= 0.5 and sum(agree)/len(agree) > 0.7:
        print("VERDICT: the event episode exists and date matching mostly finds it.")
        print("         Fix the matcher to use overlap and re-mine.")
    elif med >= 0.5:
        print("VERDICT: the event episode exists but is NOT reliably on the graph's")
        print("         date. Switch the matcher to overlap scoring and re-mine.")
    else:
        print("VERDICT: no episode matches the event in most rows. The event is")
        print("         recorded in the graph but not narrated in the dialogue.")
        print("         Mining cannot produce inference items from this dataset --")
        print("         and any earlier experiment that located 'the event' by date")
        print("         was reading an unrelated conversation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    main(a.n, a.seed, a.show)
