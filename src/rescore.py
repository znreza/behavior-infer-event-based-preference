#!/usr/bin/env python3
"""Re-parse the stored `raw` replies with the current parser. No GPU.

    python rescore.py results/generate_thinkoff__phase1_gen.csv

Writes <name>_rescored.csv beside the input and prints what changed.

"""
import argparse
import os
import sys

from parsing import extract_answer, split_reasoning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import pandas as pd
    d = pd.read_csv(a.csv)
    for col in ("raw", "opt_a", "opt_b", "correct"):
        if col not in d:
            sys.exit(f"{a.csv} has no `{col}` column -- cannot re-score. "
                     "Runs before opt_a/opt_b was added are not recoverable.")
    d = d[d.get("id", "") != "LOAD_FAILED"].copy()

    old_m = d.parse_method.copy()
    old_p = d.parsed.copy()
    old_c = d.is_correct.copy()

    # A clipped `raw` cannot be re-scored: the answer tag sits at the END of a
    # reply, and early runs stored text[:4000]. The runner parsed the FULL text
    # before storing, so its parse is the correct one -- overwriting it from a
    # truncated record would replace a good answer with a worse guess. Leave
    # those rows exactly as the run recorded them.
    if "raw_clipped" in d:
        clipped = d.raw_clipped.fillna(False).astype(bool)
    else:
        clipped = d.raw.fillna("").str.len() >= 4000
    n_clip = int(clipped.sum())

    picked, method, reason_len = [], [], []
    for r in d.itertuples():
        opts = [r.opt_a, r.opt_b]
        raw = "" if pd.isna(r.raw) else str(r.raw)
        if clipped.loc[r.Index]:
            picked.append(r.picked)
            method.append(r.parse_method)
            reason_len.append(r.n_reason_chars)
            continue
        ans, how = extract_answer(raw, opts)
        rs, _ = split_reasoning(raw)
        picked.append(ans)
        method.append(how)
        reason_len.append(len(rs))
    d["picked"] = picked
    d["parse_method"] = method
    d["parsed"] = [p is not None and str(p) != "nan" for p in picked]
    d["is_correct"] = [p == c for p, c in zip(picked, d.correct)]
    # picked_old used to be left at whatever the run recorded, so a row whose
    # answer CHANGED under rescoring kept its old stale-preference flag. On
    # phase1_gen that left Qwen2.5-7B at picked_old=27.9% against an accuracy
    # of 13.3%, which cannot both be true in a two-option choice. In T1, T2
    # and T4 the distractor IS the previously stated value (preflight asserts
    # it), so the flag is recoverable exactly; T3 never offers it.
    stale = []
    for r in d.itertuples():
        if r.condition == "T3_counterfact" or not r.parsed:
            stale.append(False)
            continue
        other = r.opt_b if r.picked == r.opt_a else r.opt_a
        stale.append(bool(r.picked != r.correct and other == r.correct))
    d["picked_old"] = stale
    d["n_reason_chars"] = reason_len
    d["rescored"] = ~clipped

    out = a.out or a.csv.replace(".csv", "_rescored.csv")
    d.to_csv(out, index=False)

    print(f"{a.csv}\n  {len(d)} rows -> {out}")
    print(f"  {len(d)-n_clip} re-scored; {n_clip} left as the run recorded them "
          f"(stored text clipped -- not re-scorable)\n")
    for m, g in d.groupby("model"):
        i = g.index
        print(f"  {m}")
        print(f"    parse rate {100*old_p[i].mean():5.1f}%  ->  "
              f"{100*g.parsed.mean():5.1f}%   "
              f"({int(g.parsed.sum() - old_p[i].sum()):+d} rows)")
        ch = g[old_m[i].values != g.parse_method.values]
        print(f"    method changed on {len(ch)} rows")
        if len(ch):
            print("      from:", dict(old_m[ch.index].value_counts().head(4)))
            print("      to  :", dict(ch.parse_method.value_counts().head(4)))
        print("    accuracy among parsed, by condition:")
        for c in sorted(g.condition.unique()):
            s = g[g.condition == c]
            o = d.loc[s.index]
            was = old_c[s.index][old_p[s.index]]
            now = s[s.parsed].is_correct
            print(f"      {c:16s} n {old_p[s.index].sum():4d}->{int(s.parsed.sum()):4d}   "
                  f"acc {100*was.mean() if len(was) else float('nan'):5.1f}% -> "
                  f"{100*now.mean() if len(now) else float('nan'):5.1f}%")
        print()


if __name__ == "__main__":
    main()
