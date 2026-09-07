#!/usr/bin/env python3
"""
Is the CAUSE of a preference change actually narrated in the dialogue?

    python check_dataset.py search --name personamem
    python check_dataset.py search --name realpref
    python check_dataset.py schema --dataset <id>
    python check_dataset.py probe  --dataset <id>

WHY
---
HorizonBench failed this test: its events are recorded in the mental state graph
but never said out loud in the conversation. Median overlap between the event
name and the best-matching episode was 0.33, and the graph's own date pointed at
the right episode 1.7% of the time. Items mined from it test nothing.

Before trusting any other dataset, run the same test. A dataset is usable for
event-to-preference inference only if the reason for a change appears in the
dialogue, in the user's own words, without stating the new preference outright.

WHAT TO LOOK FOR
----------------
  reason field present               the dataset records WHY a preference changed
  reason narrated in the dialogue    overlap between that reason and the text
  new value NOT in the event text    otherwise it is an explicit item

PersonaMem is the more promising of the two: it advertises query types for
tracking preference evolution and for revisiting the reasons behind preference
updates. A question of the form "why did this user change their mind about X"
is only answerable if the reason is in the context, so the prior is good. Verify
rather than assume.

I do not know either schema. `search` finds the dataset id, `schema` dumps the
structure, and `probe` runs the diagnostic once you have filled in FIELD_MAP.
"""

import argparse, json, re, sys
from collections import Counter

# Guesses only -- resolve with `search` before using.
CANDIDATES = {
    "personamem": ["bowen-chen/PersonaMem", "PersonaMem/PersonaMem", "personamem"],
    "realpref": ["RealPref/RealPref", "realpref"],
    "longmemeval": ["xiaowu0162/longmemeval"],
    "prefeval": ["PrefEval/PrefEval", "siyanzhao/PrefEval"],
}

# ---- fill this in after running `schema` ------------------------------------
FIELD_MAP = dict(
    conversation=None,   # the dialogue: str, or list of sessions/turns
    reason=None,         # why the preference changed (free text or event name)
    old_value=None,      # preference before
    new_value=None,      # preference after
    question=None,       # the query, if the dataset is QA-shaped
    qtype=None,          # question-type label, if present
)
# ------------------------------------------------------------------------------

STOP = set("""the a an and or of to in is are was were i you it that this for on with my me we they
he she but so if as at be been have has had do does did not no yes about from into over under
their there here what when where which who whom will would can could should than then them our
your his her its just really very much more most some any all one two user assistant""".split())

CAUSE_KEYS = ("reason", "cause", "why", "event", "trigger", "change", "update",
              "evolution", "shift", "motivation", "because")


def words(s):
    return {w for w in re.findall(r"[a-z]{4,}", str(s).lower()) if w not in STOP}


def flatten(o, limit=200000):
    """Any nested structure -> one string, for overlap scoring."""
    out = []

    def walk(x, depth=0):
        if depth > 6 or sum(len(s) for s in out) > limit:
            return
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, depth + 1)
    walk(o)
    return " ".join(out)[:limit]


# ===========================================================================

def cmd_configs(dataset, rows):
    """Dump every config's fields at once. A dataset's interesting content is
    often not in the first config -- PersonaMem-v3's `persona_context` is a raw
    activity log, while the preference-update questions live elsewhere."""
    from datasets import load_dataset, get_dataset_config_names
    cfgs = get_dataset_config_names(dataset)
    print(f"{dataset}: configs = {cfgs}\n")
    for c in cfgs:
        print("=" * 72)
        print(f"CONFIG: {c}")
        try:
            ds = load_dataset(dataset, c)
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        sp = list(ds.keys())[0]
        D = ds[sp]
        print(f"  split {sp}, {len(D)} rows")
        ex = D[0]
        for k, v in ex.items():
            s = str(v)
            print(f"    {k:30s} {type(v).__name__:9s} {s[:90]!r}")
        hits = [k for k in ex if any(c2 in k.lower() for c2 in CAUSE_KEYS)]
        if hits:
            print(f"  cause-ish fields: {hits}")
        for i in range(min(rows, len(D))):
            print(f"\n  --- {c} row {i} ---")
            print("  " + json.dumps(
                {k: (str(v)[:300] + "..." if len(str(v)) > 300 else v)
                 for k, v in D[i].items()}, indent=2, default=str)[:1600]
                  .replace("\n", "\n  "))
    print("\nPick the config that holds preference CHANGES, then run `scan` on its "
          "cause field.")


def cmd_scan(dataset, config, split, field, rows, show):
    """How often is a field actually populated, and what does it look like when
    it is? `preference_evolution` being null in the first three rows does not
    mean it is null everywhere."""
    from datasets import load_dataset
    ds = load_dataset(dataset, config) if config else load_dataset(dataset)
    sp = split if split in ds else list(ds.keys())[0]
    D = ds[sp]
    n = min(rows, len(D))
    print(f"scanning {n} of {len(D)} rows in {sp} for {field!r}\n")

    nonnull, samples = 0, []
    for i in range(n):
        v = D[i].get(field)
        empty = v in (None, "", "null", "[]", "{}", "None")
        if not empty:
            try:
                p = json.loads(v) if isinstance(v, str) else v
                if p in (None, [], {}):
                    empty = True
            except Exception:
                pass
        if not empty:
            nonnull += 1
            if len(samples) < show:
                samples.append((i, v))

    print(f"populated: {nonnull}/{n}  ({100*nonnull/max(n,1):.1f}%)")
    if not nonnull:
        print("\n>>> the field is empty in this sample. Either it is unused in this")
        print("    config, or it is populated only in rows you have not sampled.")
        print("    Try --rows 20000, or a different config.")
        return
    print("\n=== populated examples ===")
    for i, v in samples:
        txt = json.dumps(json.loads(v), indent=2)[:1200] if isinstance(v, str) and \
              v.strip().startswith(("{", "[")) else str(v)[:1200]
        print(f"\n--- row {i} ---\n{txt}")


def load_local(path):
    """RealPref and similar ship on GitHub, not the Hub. Point at a cloned repo
    directory or a single file; json / jsonl / csv are read."""
    import os, csv as _csv
    files = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            if any(p in root for p in (".git", "__pycache__")):
                continue
            for nm in names:
                if nm.endswith((".json", ".jsonl", ".csv")):
                    files.append(os.path.join(root, nm))
    else:
        files = [path]
    if not files:
        sys.exit(f"no json/jsonl/csv under {path}")

    print(f"found {len(files)} data files under {path}:")
    for f in sorted(files)[:40]:
        print(f"   {os.path.getsize(f):>12,d}  {os.path.relpath(f, path)}")
    if len(files) > 40:
        print(f"   ... and {len(files)-40} more")

    biggest = max(files, key=os.path.getsize)
    print(f"\nreading the largest: {os.path.relpath(biggest, path)}")
    rows = []
    if biggest.endswith(".jsonl"):
        with open(biggest) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                if len(rows) >= 2000:
                    break
    elif biggest.endswith(".json"):
        obj = json.load(open(biggest))
        rows = obj if isinstance(obj, list) else [obj]
    else:
        with open(biggest) as fh:
            rows = list(_csv.DictReader(fh))[:2000]
    print(f"loaded {len(rows)} rows")
    return rows


def cmd_local(path, show):
    rows = load_local(path)
    if not rows:
        sys.exit("no rows")
    ex = rows[0]
    print("\n=== FIELDS ===")
    for k, v in (ex.items() if isinstance(ex, dict) else []):
        s = str(v)
        print(f"  {k:32s} {type(v).__name__:9s} len={len(s):8d}  {s[:90]!r}")
    if isinstance(ex, dict):
        hits = [k for k in ex if any(c in k.lower() for c in CAUSE_KEYS)]
        print(f"\ncause-ish fields: {hits if hits else 'none by name'}")
    print("\n=== SAMPLE ROWS ===")
    for i, r in enumerate(rows[:show]):
        print(f"\n--- row {i} ---")
        print(json.dumps(r, indent=2, default=str)[:2200])
    print("\nFill FIELD_MAP, then run: probe --path <same path>")


def probe_rows(rows, F):
    """Shared probe body, so Hub and local data go through identical scoring."""
    import statistics as st
    scores, leaks, checked, examples = [], 0, 0, []
    for r in rows:
        reason, conv = r.get(F["reason"]), r.get(F["conversation"])
        if not reason or not conv:
            continue
        q = words(flatten(reason))
        if len(q) < 2:
            continue
        hay = words(flatten(conv))
        s = len(q & hay) / len(q)
        scores.append(s)
        nv = r.get(F["new_value"]) if F.get("new_value") else None
        if nv:
            nvw = str(flatten(nv)).lower()
            if len(nvw) > 3 and nvw in flatten(conv).lower():
                leaks += 1
        checked += 1
        if len(examples) < 5:
            examples.append((flatten(reason)[:220], s))
    return scores, leaks, checked, examples


def report(scores, leaks, checked, examples, has_new_value):
    import statistics as st
    if not scores:
        sys.exit("no usable rows -- check FIELD_MAP against the schema dump")
    print(f"\nchecked {checked} rows")
    print("overlap between the stated reason and the dialogue:")
    print(f"   median {st.median(scores):.2f}   mean {sum(scores)/len(scores):.2f}")
    b = Counter("0.0-0.2" if s < .2 else "0.2-0.4" if s < .4 else
                "0.4-0.6" if s < .6 else "0.6-0.8" if s < .8 else "0.8-1.0"
                for s in scores)
    for k in ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"):
        c = b.get(k, 0)
        print(f"   {k}  {c:4d}  {'#' * int(40*c/max(checked,1))}")
    if has_new_value:
        print(f"\ndialogue literally contains the NEW value: "
              f"{100*leaks/checked:.1f}% of rows")
    print("\n=== sample reasons ===")
    for txt, s in examples:
        print(f"  [{s:.2f}] {txt}")
    med = st.median(scores)
    print()
    if med >= 0.6:
        print("VERDICT: the cause is narrated in the dialogue. Usable -- next is a")
        print("         human annotation pass on ~40 items.")
    elif med >= 0.4:
        print("VERDICT: partial. Filter on per-row overlap, keep the top band, and")
        print("         annotate that band before using it.")
    else:
        print("VERDICT: the cause is not in the dialogue, same as HorizonBench.")
        print("         Not usable for event-to-preference inference.")




def _parse_where(w):
    if not w:
        return None, None
    if "=" not in w:
        sys.exit("--where must look like field=value")
    f, v = w.split("=", 1)
    return f.strip(), v.strip()


def _apply_where(D, where, limit=None):
    """Return a list of row dicts matching field=value."""
    f, v = _parse_where(where)
    out = []
    for i in range(len(D)):
        r = D[i]
        if f is None or str(r.get(f)) == v:
            out.append(r)
        if limit and len(out) >= limit:
            break
    return out



def cmd_rows(dataset, config, split, where, show, fields):
    """Print full rows matching field=value. Use this to read a rare task type
    -- e.g. the 62 `preference_shift_followthrough` items -- rather than judging
    a 15k-row config from its first two rows."""
    from datasets import load_dataset
    ds = load_dataset(dataset, config) if config else load_dataset(dataset)
    sp = split if split in ds else list(ds.keys())[0]
    D = ds[sp]
    rows = _apply_where(D, where)
    print(f"{len(rows)} rows match {where!r} out of {len(D)}\n")
    keep = [f.strip() for f in fields.split(",")] if fields else None
    for i, r in enumerate(rows[:show]):
        print("=" * 72)
        print(f"--- match {i} ---")
        d = {k: v for k, v in r.items() if (keep is None or k in keep)}
        print(json.dumps({k: (str(v)[:1500] + " ...[truncated]"
                              if len(str(v)) > 1500 else v)
                          for k, v in d.items()}, indent=2, default=str))
    if len(rows) > show:
        print(f"\n({len(rows)-show} more matches not shown)")


def cmd_values(dataset, config, split, field, rows):
    """Distribution of a categorical field. Two sample rows tell you nothing
    about which task types exist across 15k rows."""
    from datasets import load_dataset
    ds = load_dataset(dataset, config) if config else load_dataset(dataset)
    sp = split if split in ds else list(ds.keys())[0]
    D = ds[sp]
    n = min(rows, len(D))
    c = Counter(str(D[i].get(field)) for i in range(n))
    print(f"{field!r} over {n} of {len(D)} rows in {sp}:\n")
    for v, k in c.most_common(40):
        print(f"  {k:7d}  {100*k/n:5.1f}%  {v[:100]}")
    if len(c) > 40:
        print(f"  ... and {len(c)-40} more distinct values")


def cmd_search(name):
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"searching the Hub for {name!r}\n")
    hits = list(api.list_datasets(search=name, limit=30))
    if not hits:
        print("no results. Try a different spelling, or find the id in the paper "
              "or its GitHub repo.")
    for d in hits:
        dl = getattr(d, "downloads", None)
        print(f"  {d.id:60s} downloads={dl}")
    print("\nAlso try the paper's GitHub -- many of these ship data outside the Hub.")
    if name.lower() in CANDIDATES:
        print(f"\nguesses on file (unverified): {CANDIDATES[name.lower()]}")


def cmd_schema(dataset, config, split, n):
    from datasets import load_dataset, get_dataset_config_names
    try:
        cfgs = get_dataset_config_names(dataset)
        print(f"configs: {cfgs}")
    except Exception as e:
        print(f"(config listing failed: {e})")
        cfgs = []
    ds = load_dataset(dataset, config) if config else load_dataset(dataset)
    print(ds)
    sp = split if split in ds else list(ds.keys())[0]
    D = ds[sp]
    ex = D[0]

    print(f"\n=== TOP-LEVEL FIELDS ({sp}, {len(D)} rows) ===")
    for k, v in ex.items():
        s = str(v)
        print(f"{k:34s} {type(v).__name__:10s} len={len(s):9d}  {s[:110]!r}")

    print("\n=== FIELDS THAT MIGHT HOLD A CAUSE ===")
    hits = [k for k in ex if any(c in k.lower() for c in CAUSE_KEYS)]
    print("  ", hits if hits else "none by name -- check nested structures below")

    print("\n=== NESTED STRUCTURE ===")

    def walk(o, prefix="", depth=0):
        if depth > 3:
            return
        if isinstance(o, dict):
            for k, v in list(o.items())[:12]:
                print("  " * depth + f"{prefix}{k}: {type(v).__name__}")
                walk(v, "", depth + 1)
        elif isinstance(o, list) and o:
            print("  " * depth + f"[list len={len(o)}] first:")
            walk(o[0], "", depth + 1)
    walk(ex)

    print("\n=== SAMPLE ROWS ===")
    for i in range(min(n, len(D))):
        print(f"\n--- row {i} ---")
        print(json.dumps({k: (str(v)[:400] + "..." if len(str(v)) > 400 else v)
                          for k, v in D[i].items()}, indent=2, default=str)[:2200])

    print("\nNext: fill FIELD_MAP at the top of this file, then run `probe`.")


def cmd_probe(dataset, config, split, n_rows, seed, where=None):
    import random
    from datasets import load_dataset
    if not FIELD_MAP.get("conversation") or not FIELD_MAP.get("reason"):
        sys.exit("fill FIELD_MAP['conversation'] and FIELD_MAP['reason'] first "
                 "(run `schema` to see the field names)")

    ds = load_dataset(dataset, config) if config else load_dataset(dataset)
    sp = split if split in ds else list(ds.keys())[0]
    D = ds[sp]
    if where:
        sub = _apply_where(D, where)
        print(f"filtered to {len(sub)} rows matching {where!r}")
        sc, lk, ch, ex = probe_rows(sub, FIELD_MAP)
        report(sc, lk, ch, ex, bool(FIELD_MAP.get("new_value")))
        return

    rng = random.Random(seed)
    idx = list(range(len(D)))
    rng.shuffle(idx)

    F = FIELD_MAP
    scores, leaks, checked = [], 0, 0
    examples = []
    for i in idx:
        if checked >= n_rows:
            break
        r = D[i]
        reason = r.get(F["reason"])
        conv = r.get(F["conversation"])
        if not reason or not conv:
            continue
        q = words(flatten(reason))
        if len(q) < 2:
            continue
        hay = words(flatten(conv))
        s = len(q & hay) / len(q)
        scores.append(s)

        # does the dialogue also state the new value outright?
        nv = r.get(F["new_value"]) if F.get("new_value") else None
        if nv:
            nvw = str(flatten(nv)).lower()
            if len(nvw) > 3 and nvw in flatten(conv).lower():
                leaks += 1
        checked += 1
        if len(examples) < 5:
            examples.append((flatten(reason)[:220], s))

    if not scores:
        sys.exit("no usable rows -- check FIELD_MAP against the schema dump")

    import statistics as st
    print(f"\nchecked {checked} rows")
    print(f"overlap between the stated reason and the dialogue:")
    print(f"   median {st.median(scores):.2f}   mean {sum(scores)/len(scores):.2f}")
    b = Counter("0.0-0.2" if s < .2 else "0.2-0.4" if s < .4 else
                "0.4-0.6" if s < .6 else "0.6-0.8" if s < .8 else "0.8-1.0"
                for s in scores)
    for k in ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"):
        c = b.get(k, 0)
        print(f"   {k}  {c:4d}  {'#' * int(40*c/max(checked,1))}")
    if FIELD_MAP.get("new_value"):
        print(f"\ndialogue literally contains the NEW value: "
              f"{100*leaks/checked:.1f}% of rows")
        print("   (high here means many items are explicit, not inference)")

    print("\n=== sample reasons ===")
    for txt, s in examples:
        print(f"  [{s:.2f}] {txt}")

    med = st.median(scores)
    print()
    if med >= 0.6:
        print("VERDICT: the cause is narrated in the dialogue. Usable -- next step")
        print("         is a human annotation pass on a sample of ~40 items.")
    elif med >= 0.4:
        print("VERDICT: partial. Some rows narrate the cause and some do not.")
        print("         Filter on the per-row overlap score, keep the top band,")
        print("         and annotate that band before using it.")
    else:
        print("VERDICT: the cause is not in the dialogue, same as HorizonBench.")
        print("         Not usable for event-to-preference inference.")


# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "schema", "configs", "scan",
                                     "values", "rows", "local", "probe"])
    ap.add_argument("--name", default="personamem")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--rows", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--field", default="preference_evolution")
    ap.add_argument("--path", default="", help="local repo dir or file (RealPref)")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--where", default="", help="field=value row filter")
    ap.add_argument("--fields", default="", help="comma-separated fields to print")
    a = ap.parse_args()

    if a.mode == "search":
        cmd_search(a.name)
    elif a.mode == "configs":
        if not a.dataset:
            sys.exit("--dataset required")
        cmd_configs(a.dataset, a.n)
    elif a.mode == "scan":
        if not a.dataset:
            sys.exit("--dataset required")
        cmd_scan(a.dataset, a.config or None, a.split, a.field, a.rows, a.show)
    elif a.mode == "values":
        if not a.dataset:
            sys.exit("--dataset required")
        cmd_values(a.dataset, a.config or None, a.split, a.field, a.rows)
    elif a.mode == "rows":
        if not a.dataset:
            sys.exit("--dataset required")
        cmd_rows(a.dataset, a.config or None, a.split, a.where, a.show, a.fields)
    elif a.mode == "local":
        if not a.path:
            sys.exit("--path required")
        cmd_local(a.path, a.show)
    elif a.mode == "schema":
        if not a.dataset:
            sys.exit("--dataset required (run `search` first)")
        cmd_schema(a.dataset, a.config or None, a.split, a.n)
    else:
        if not FIELD_MAP.get("conversation") or not FIELD_MAP.get("reason"):
            sys.exit("fill FIELD_MAP['conversation'] and FIELD_MAP['reason'] first")
        if a.path:
            rows = load_local(a.path)
            sc, lk, ch, ex = probe_rows(rows, FIELD_MAP)
            report(sc, lk, ch, ex, bool(FIELD_MAP.get("new_value")))
        else:
            if not a.dataset:
                sys.exit("--dataset or --path required")
            cmd_probe(a.dataset, a.config or None, a.split, a.rows, a.seed,
                      a.where or None)
