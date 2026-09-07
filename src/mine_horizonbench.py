#!/usr/bin/env python3
"""
Derive items from HorizonBench's released mental state graph.

    python mine_horizonbench.py --n 400 --out hb_items.json
    python mine_horizonbench.py --n 400 --out hb_items.json --inspect

WHY THIS EXISTS
---------------
The hand-written stimuli answer "does the model infer a preference change from
its cause" cleanly, but a reviewer can fairly ask where the items came from and
whether they measure the intended thing at all. Human annotation answers the
second question. This script answers the first.

HorizonBench (arXiv:2604.17283) already contains the triple we need, generated
and filtered by its own pipeline:

    changed_attributes   {attr: {original, current}}
    evolution_history    [{date, event_name, event_category, attribute_changes}]
    conversation         the dated episode where the event was mentioned

So the causal claim -- that this event implies this preference change -- is
theirs, not ours.

WHAT IS BORROWED AND WHAT IS CONTROLLED
---------------------------------------
  borrowed    the event episode text, verbatim from their conversation
              the old and new attribute values
              the attribute name
  controlled  the setup sentence (we state the old value plainly)
              the filler turns
              the question and the two options
              the length: four short conversations, not 163K tokens

That split is deliberate. The thing under test is the inference from the event,
so the event comes from an external source. Everything that could confound the
measurement -- length, retrieval difficulty, option asymmetry -- is ours to hold
fixed. Items are emitted in the same schema as build_items() in
the item generator, so the two sets run through the same code.

CAVEAT worth stating in the paper: HorizonBench is itself LLM-generated. Using
it removes "the authors invented these" but not "these are synthetic". The human
annotation pass covers what remains.
"""

import argparse, json, re, random, sys
from collections import Counter, defaultdict

DATASET = "stellalisy/HorizonBench"
EP_RE = re.compile(r"^Date:\s*(\S+)\s*\nScenario:\s*(.+)$", re.M)
USER_RE = re.compile(r"^([A-Z][a-z]+):\s*(.+)$", re.M)


def humanize(v):
    """bullet_point_action_steps -> bullet point action steps"""
    return str(v).replace("_", " ").strip()


def split_episodes(conv):
    hits = list(EP_RE.finditer(conv))
    out = []
    for k, h in enumerate(hits):
        start = h.start()
        end = hits[k + 1].start() if k + 1 < len(hits) else len(conv)
        out.append(dict(date=h.group(1), scenario=h.group(2).strip(),
                        text=conv[start:end].strip(), idx=k))
    return out


def first_user_turns(ep_text, max_turns=1, max_chars=400):
    """The user's OPENING turn only, by default.

    Two reasons to take just one. The assistant's reply often paraphrases the
    implication and would hand over the answer. And later user turns in the same
    episode frequently drift into stating the new preference outright -- e.g.
    "everything fell apart ... I don't want gentle noises right now, I need to
    know what to actually do" -- which turns an inference item into an explicit
    one. The literal-value filter downstream will not catch that, because the
    stored value is something like `firm_directness` and the dialogue says it in
    ordinary words. Only human annotation catches it reliably; taking one turn
    reduces how often it happens."""
    lines = [l for l in ep_text.split("\n")
             if l.strip() and not l.startswith(("Date:", "Scenario:"))]
    turns = []
    for l in lines:
        m = USER_RE.match(l.strip())
        if not m:
            continue
        speaker, content = m.group(1), m.group(2)
        if speaker.lower() == "assistant":
            if turns:
                break                      # stop at the first assistant reply
            continue
        turns.append(content.strip())
        if len(turns) >= max_turns:
            break
    text = " ".join(turns)[:max_chars]
    return text.strip()


def mine(n_target, seed, inspect=False, max_rows=1500):
    from datasets import load_dataset
    ds = load_dataset(DATASET)
    split = "test" if "test" in ds else list(ds.keys())[0]
    D = ds[split]
    print(f"{DATASET}: {len(D)} rows")

    rng = random.Random(seed)
    idx = [i for i, h in enumerate(D["has_evolved"]) if h]
    rng.shuffle(idx)
    idx = idx[:max_rows]

    items, stats = [], Counter()
    seen = set()
    for i in idx:
        if len(items) >= n_target:
            break
        r = D[i]
        try:
            pe = json.loads(r["preference_evolution"])
        except Exception:
            stats["no_evolution_json"] += 1
            continue
        chg = pe.get("changed_attributes") or {}
        hist = pe.get("evolution_history") or []
        if not chg or not hist:
            stats["no_change_or_history"] += 1
            continue

        eps = split_episodes(r["conversation"])
        if len(eps) < 3:
            stats["too_few_episodes"] += 1
            continue

        for h in hist:
            ev_date = str(h.get("date", ""))[:10]
            ev_name = h.get("event_name", "")
            same_day = [e for e in eps if str(e["date"])[:10] == ev_date]
            if not same_day:
                stats["event_episode_not_found"] += 1
                continue
            if len(same_day) > 1:
                q = set(re.findall(r"[a-z]{4,}", ev_name.lower()))
                same_day.sort(
                    key=lambda e: -len(q & set(re.findall(r"[a-z]{4,}",
                                                          e["scenario"].lower()))))
            ep = same_day[0]
            ev_text = first_user_turns(ep["text"])
            if len(ev_text) < 30:
                stats["event_text_too_short"] += 1
                continue

            for attr, ch in (h.get("attribute_changes") or chg).items():
                old = ch.get("from", ch.get("original"))
                new = ch.get("to", ch.get("current"))
                if not old or not new or old == new:
                    continue
                old_h, new_h = humanize(old), humanize(new)

                # drop items where the event text literally contains either
                # value: those test copying, not inference
                low = ev_text.lower()
                if any(w in low for w in (old_h.lower(), new_h.lower())):
                    stats["lexical_leak_dropped"] += 1
                    continue

                key = (attr, old, new, ev_name)
                if key in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(key)

                attr_h = humanize(attr)
                items.append(dict(
                    id=f"hb_{len(items):04d}",
                    source="horizonbench",
                    domain=attr,
                    noun=attr_h,
                    direction="fwd",
                    old=old_h, new=new_h,
                    event=ev_text,
                    event_name=ev_name,
                    event_category=h.get("event_category", ""),
                    user_id=r.get("user_id", ""),
                    setup=f"For {attr_h}, what works for me is {old_h}.",
                    fillers=None,          # filled in at render time
                ))
                stats["kept"] += 1
                if len(items) >= n_target:
                    break
            if len(items) >= n_target:
                break

    print("\nmining stats:", dict(stats))
    print(f"kept {len(items)} items over {len({i['domain'] for i in items})} attributes")
    print("\nattributes:")
    for a, c in Counter(i["domain"] for i in items).most_common(15):
        print(f"   {c:4d}  {a}")
    print("\nevent categories:")
    for a, c in Counter(i["event_category"] for i in items).most_common(10):
        print(f"   {c:4d}  {a}")

    if inspect:
        print("\n" + "=" * 70)
        for it in items[:6]:
            print(f"\n[{it['id']}]  attribute: {it['noun']}")
            print(f"  old -> new : {it['old']}  ->  {it['new']}")
            print(f"  event_name : {it['event_name']}")
            print(f"  event text : {it['event'][:300]}")
        print("\n" + "=" * 70)
        print("READ THESE for two failure modes:")
        print("  1. the event does not plainly imply the change -> unusable")
        print("  2. the user STATES the new preference in ordinary words -> this")
        print("     is an explicit item, not an inference item, and will inflate")
        print("     accuracy. The literal-value filter cannot catch it.")
        print("Both are what the human annotation pass is for. Do not run models")
        print("on mined items before annotating a sample.")

    return items


def add_fillers(items, seed=0):
    """Attach filler turns and names so these render through the same
    _history()/build_question() code as the hand-written set."""
    # The v1 runner these came from is in _archive/. NAMES is unchanged;
    # the filler pool grew from 20 to 58 after v1, so a re-run will not
    # reproduce the archived hb_items.json byte for byte.
    from generate_v2 import NAMES                  # no modal import at module load
    from modal_runner import FILLER
    rng = random.Random(seed)
    for it in items:
        it["name"] = rng.choice(NAMES)
        it["fillers"] = rng.sample(FILLER, 2)
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="hb_items.json")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    items = mine(a.n, a.seed, a.inspect)
    if not items:
        sys.exit("no items mined -- check the schema against the released data")
    with open(a.out, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")
    print("next: sample ~40 of these into the annotation pass before running any "
          "model on them.")
