#!/usr/bin/env python3
"""How often does a filler turn touch the topic under test, and does it matter?

    python check_fillers.py

A filler is meant to be inert padding. None contains an option label verbatim,
which is asserted at build time, but that check is too weak: a filler need not
name an option to change how a reader reasons. One annotated item was answered
partly from a filler about a leaking kitchen in a meal-preparation item, which
is what prompted this pass.

The script reports, for every one of the 27,720 items, whether any filler it
displays is topically adjacent to the item's attribute, and then tests whether
adjacency moves accuracy. Assignment of fillers is random with respect to the
keyed answer, so adjacency should add noise and not bias; the second half of
the output is what checks that rather than assuming it.
"""
import json
import math
import random
import re
from collections import Counter

import pandas as pd

import generate_v2 as G

TOPIC = {
    "transport": r"car\b|drive|driving|bicycl|bike|cycl|bus\b|train\b|commut|"
                 r"parking|petrol|fuel|road",
    "exercise": r"gym|running|swim|walk|workout|fitness|knee|shoulder",
    "mealprep": r"kitchen|cook|meal|recipe|oven|stove|fridge|freezer|grocer|"
                r"food",
    "shopping": r"shop|buying|purchase|store\b|market|delivery|order\b",
    "travel": r"trip|holiday|flight|hotel|travel|abroad",
    "music": r"music|song|band\b|album|playlist|guitar",
    "reading": r"book\b|books\b|reading|novel|library",
    "learning": r"learn|course|study|tutorial|revis",
    "communication": r"email|text message|texting|phone|call\b|calling|message",
    "replylength": r"reply|replies|answer|respond|explain",
    "replytone": r"reply|replies|tone|polite|blunt",
    "replyformat": r"reply|replies|format|bullet|list\b|table",
}


def fillers_for(item):
    """Reproduce exactly the fillers this item displays."""
    rng = random.Random(item["seed"])
    n = max(4, item["n_turns"])
    pool = G.FILLER[:]
    rng.shuffle(pool)
    need = n
    fills = ([pool[i] for i in range(min(need, len(pool)))]
             + [pool[i % len(pool)] for i in range(len(pool), need)])
    ev = max(2, min(n - 1, round(item["event_pos"] * n)))
    out, fi = [], 0
    for t in range(1, n + 1):
        if t == 1:
            continue
        if t == ev and item["event_text"] is not None:
            continue
        out.append(fills[fi % len(fills)])
        fi += 1
    return out


def main():
    items = json.load(open("data/items_v2.json"))
    pat = {a: re.compile(p, re.I) for a, p in TOPIC.items()}
    flag, per_attr = {}, Counter()
    n_adj = 0
    for it in items:
        hits = [f for f in fillers_for(it) if pat[it["attribute"]].search(f)]
        flag[it["id"]] = bool(hits)
        if hits:
            n_adj += 1
            per_attr[it["attribute"]] += 1
    print(f"{len(items)} items, {n_adj} ({100*n_adj/len(items):.1f}%) display "
          f"at least one filler topically adjacent to their attribute\n")
    tot = Counter(i["attribute"] for i in items)
    print(f"{'attribute':15s} {'items':>6s} {'adjacent':>9s}")
    for a in sorted(tot):
        print(f"{a:15s} {tot[a]:6d} {100*per_attr[a]/tot[a]:8.1f}%")

    print("\nDoes adjacency move accuracy? Probe run, per condition.")
    d = pd.read_csv("results/probe_chat_prefill__phase1_main.csv",
                    low_memory=False)
    d["adj"] = d.id.map(flag)
    print(f"{'condition':16s} {'adjacent':>9s} {'not':>8s} {'diff':>7s} {'n adj':>7s}")
    for c in sorted(d.condition.unique()):
        s = d[d.condition == c]
        a = 100 * s[s.adj].is_correct.mean()
        b = 100 * s[~s.adj].is_correct.mean()
        print(f"{c:16s} {a:8.1f}% {b:7.1f}% {a-b:+7.1f} {int(s.adj.sum()):7d}")
    print("\nA filler is assigned at random with respect to the keyed answer, "
          "so a\nnon-zero difference here is noise unless it is systematic "
          "across conditions.")


if __name__ == "__main__":
    main()
