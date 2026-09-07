#!/usr/bin/env python3
"""
RealPref: what does it actually contain?

    git clone https://github.com/GG14127/RealPref
    python check_realpref.py --path RealPref/test_data

Three questions, in order of importance to us:

  1. Does any preference SUPERSEDE an earlier one (old -> new)?
     HorizonBench had this and never narrated it. PersonaMem had expiry with no
     replacement. If RealPref has no supersession either, then no public
     benchmark supplies the construct and constructed stimuli are justified.

  2. Is the Pref_init event NARRATED in the conversations?
     Pref_init records the experience that created a preference. If that text
     appears in the dialogue, RealPref can support an "infer the preference from
     the experience" task -- acquisition rather than change, but a real
     inference task with external provenance.

  3. How are expression levels 1-4 distributed and what do they look like?
     Levels run explicit to implicit. That is our E1-E4 ladder in someone else's
     data, and it can ground the continuum even if the change construct cannot
     be sourced here.
"""

import argparse, json, os, re, sys
from collections import Counter, defaultdict

STOP = set("""the a an and or of to in is are was were i you it that this for on with my me we they
he she but so if as at be been have has had do does did not no yes about from into over under
their there here what when where which who whom will would can could should than then them our
your his her its just really very much more most some any all one two user assistant she her""".split())


def words(s):
    return {w for w in re.findall(r"[a-z]{4,}", str(s).lower()) if w not in STOP}


def flatten(o, limit=400000):
    out = []

    def walk(x, d=0):
        if d > 8 or sum(len(s) for s in out) > limit:
            return
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v, d + 1)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, d + 1)
    walk(o)
    return " ".join(out)[:limit]



def sessions_of(evc):
    """[(stage, session_id, text)] -- one entry per SESSION.

    Flattening the whole event_conversation.json into one string and scoring
    against that measures topical overlap across everything the user ever said.
    It cannot tell you whether one specific session narrates one specific
    preference-forming event. Score sessions individually."""
    out = []
    if not isinstance(evc, dict):
        return out
    for stage, sess in evc.items():
        if not isinstance(sess, dict):
            continue
        for sid, turns in sess.items():
            out.append((str(stage), str(sid), flatten(turns)))
    return out


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main(root, max_users, show):
    udir = os.path.join(root, "user")
    if not os.path.isdir(udir):
        udir = root
    uids = sorted([d for d in os.listdir(udir)
                   if os.path.isdir(os.path.join(udir, d))])
    if not uids:
        sys.exit(f"no user directories under {udir}")
    print(f"{len(uids)} users under {udir}\n")
    uids = uids[:max_users]

    change_hits, init_scores, level_counts = [], [], Counter()
    leak_scores, leak_examples = [], []
    session_matched = Counter()
    pref_types, topics = Counter(), Counter()
    examples, level_examples = [], {}
    n_prefs = 0

    for uid in uids:
        d = os.path.join(udir, uid)
        prof = load(os.path.join(d, f"data_{uid}_profile.json"))
        evc = load(os.path.join(d, f"data_{uid}_event_conversation.json"))
        prc = load(os.path.join(d, f"data_{uid}_pref_conversation.json"))
        if prof is None:
            continue

        # ---- 1. is there any supersession anywhere in the profile? ----------
        blob = json.dumps(prof).lower()
        for pat in ("no longer", "used to", "previously preferred", "changed from",
                    "instead of what", "shifted from", "old preference",
                    "former preference", "replaced by", "superseded"):
            if pat in blob:
                change_hits.append((uid, pat))

        # ---- 2. is the Pref_init event narrated in the dialogue? ------------
        pinit = prof.get("Pref_init") or {}
        sess = sessions_of(evc)
        sess_words = [(st, sid, words(txt), txt) for st, sid, txt in sess]
        for pid, rec in pinit.items():
            n_prefs += 1
            ev = rec.get("Event", "")
            q = words(ev)
            if len(q) < 3 or not sess_words:
                continue

            # best-matching SESSION for this preference's init event
            best = max(sess_words, key=lambda t: len(q & t[2]) / len(q))
            s = len(q & best[2]) / len(q)
            init_scores.append(s)
            session_matched[(uid, f"{best[0]}/{best[1]}")] += 1
            if len(examples) < show:
                examples.append((uid, pid, rec.get("Preference", "")[:110],
                                 ev[:150], s, f"{best[0]}/{best[1]}",
                                 best[3][:200]))

            # 2b: does THAT session also state the preference itself?
            pref_txt = rec.get("Preference", "")
            pq = words(pref_txt)
            if len(pq) >= 3:
                ls = len(pq & best[2]) / len(pq)
                leak_scores.append(ls)
                if len(leak_examples) < 60:
                    leak_examples.append((uid, pid, pref_txt, best[3][:230], ls))

        for rec in (prof.get("Preference") or []):
            pref_types[str(rec.get("Type"))] += 1
            topics[str(rec.get("Topic")).split(":")[0]] += 1

        # ---- 3. expression levels ------------------------------------------
        if isinstance(prc, dict):
            for pid, lv in prc.items():
                if not isinstance(lv, dict):
                    continue
                for level, sessions in lv.items():
                    level_counts[str(level)] += 1
                    if str(level) not in level_examples:
                        txt = flatten(sessions)[:280]
                        level_examples[str(level)] = txt

    # =====================================================================
    print("=" * 72)
    print("1. SUPERSESSION (does a preference replace an earlier one?)")
    if change_hits:
        print(f"   {len(change_hits)} phrase hits across {len(uids)} users:")
        for uid, pat in change_hits[:12]:
            print(f"      user {uid}: {pat!r}")
        print("   >>> inspect these by hand -- they may be phrasing, not structure")
    else:
        print(f"   no supersession language in {len(uids)} user profiles.")
        print("   >>> RealPref models how preferences ARE ACQUIRED, not how they")
        print("       change. Same verdict as the others for the change construct.")

    print("\n" + "=" * 72)
    print("2. IS THE PREF_INIT EVENT NARRATED IN THE DIALOGUE?")
    if not init_scores:
        print("   no Pref_init records parsed -- check the file names")
    else:
        import statistics as st
        print(f"   {len(init_scores)} preferences over {len(uids)} users")
        print(f"   overlap of the init event with the conversations:")
        print(f"      median {st.median(init_scores):.2f}   "
              f"mean {sum(init_scores)/len(init_scores):.2f}")
        b = Counter("0.0-0.2" if s < .2 else "0.2-0.4" if s < .4 else
                    "0.4-0.6" if s < .6 else "0.6-0.8" if s < .8 else "0.8-1.0"
                    for s in init_scores)
        for k in ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"):
            c = b.get(k, 0)
            print(f"      {k}  {c:4d}  {'#' * int(40*c/max(len(init_scores),1))}")
        med = st.median(init_scores)
        if med >= 0.5:
            print("   >>> the init event IS narrated. Usable for an ACQUISITION")
            print("       inference task with external provenance.")
        else:
            print("   >>> the init event is recorded but not narrated, same")
            print("       failure as HorizonBench.")
        print("\n   examples:")
        for uid, pid, pref, ev, s, sess_id, sess_txt in examples:
            print(f"\n   [{s:.2f}] user {uid} {pid}  -> session {sess_id}")
            print(f"      pref   : {pref}")
            print(f"      event  : {ev}")
            print(f"      session: {sess_txt}")

        # diversity: if many preferences map to the SAME session, the match is
        # topical drift, not narration
        by_user = defaultdict(set)
        for (uid, sid), _ in session_matched.items():
            by_user[uid].add(sid)
        dup = sum(session_matched.values()) - len(session_matched)
        print(f"\n   distinct sessions matched: {len(session_matched)} "
              f"for {len(init_scores)} preferences "
              f"({dup} preferences share a session with another)")
        if dup > 0.4 * len(init_scores):
            print("   >>> many preferences map to the same session. That is")
            print("       topical overlap, not one session per event.")

    print("\n" + "=" * 72)
    print("2b. DOES THE EVENT CONVERSATION GIVE THE PREFERENCE AWAY?")
    print("    An overlap of 0.85 is high enough to worry about. If the event")
    print("    conversation also states the preference, this is retrieval, not")
    print("    inference -- the trap that killed the mined HorizonBench items.")
    if not leak_scores:
        print("    no event conversations parsed")
    else:
        import statistics as st2
        print(f"    {len(leak_scores)} preferences")
        print(f"    overlap of the PREFERENCE text with the EVENT conversation:")
        print(f"       median {st2.median(leak_scores):.2f}   "
              f"mean {sum(leak_scores)/len(leak_scores):.2f}")
        bl = Counter("0.0-0.2" if s < .2 else "0.2-0.4" if s < .4 else
                     "0.4-0.6" if s < .6 else "0.6-0.8" if s < .8 else "0.8-1.0"
                     for s in leak_scores)
        for k in ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"):
            c = bl.get(k, 0)
            print(f"       {k}  {c:4d}  {'#' * int(40*c/max(len(leak_scores),1))}")
        ml = st2.median(leak_scores)
        if ml < 0.4:
            print("    >>> GOOD: the event conversation describes the experience")
            print("        without naming the preference. A real inference task.")
        elif ml < 0.6:
            print("    >>> MIXED: filter per item on this score and keep the low")
            print("        band before using them.")
        else:
            print("    >>> the preference is largely restated in the event")
            print("        conversation. These are retrieval items, not inference.")
        print("\n    highest-leak examples (read these):")
        for uid, pid, pref, evtxt, s in sorted(leak_examples,
                                               key=lambda x: -x[4])[:3]:
            print(f"\n    [{s:.2f}] user {uid} {pid}")
            print(f"       pref     : {pref[:130]}")
            print(f"       event conv: {evtxt[:230]}")

    print("\n" + "=" * 72)
    print("3. EXPRESSION LEVELS (1 explicit -> 4 implicit)")
    if not level_counts:
        print("   none parsed")
    else:
        tot = sum(level_counts.values())
        for lv in sorted(level_counts):
            c = level_counts[lv]
            print(f"   level {lv}: {c:5d}  ({100*c/tot:.1f}%)")
        print("\n   sample of each level:")
        for lv in sorted(level_examples):
            print(f"\n   --- level {lv} ---")
            print(f"   {level_examples[lv][:260]}")
        print("\n   >>> if these read explicit -> implicit, this grounds the")
        print("       E1-E4 continuum in an external dataset even though the")
        print("       change construct has to stay hand-built.")

    print("\n" + "=" * 72)
    print("preference Types:", dict(pref_types))
    print("topics:", dict(topics.most_common(10)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="RealPref/test_data")
    ap.add_argument("--users", type=int, default=25)
    ap.add_argument("--show", type=int, default=4)
    a = ap.parse_args()
    main(a.path, a.users, a.show)
