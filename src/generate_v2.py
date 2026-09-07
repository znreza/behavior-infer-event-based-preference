#!/usr/bin/env python3
"""
v2 stimulus generator: matched condition sets from author-written anchors.

    python generate_v2.py --anchors anchors.json --turns 4,10,20,50 --out data/items_v2.json
    python generate_v2.py --anchors anchors.json --inspect

WHAT IS AUTHORED AND WHAT IS GENERATED
--------------------------------------
The causal claim -- this old state, plus this event, gives this new state -- is
written by hand in the anchors file. The generator only renders it into a
conversation. So the ground truth is not an LLM's judgement about what counts as
a valid inference; that was the main bias in v1, where the answer key came from
the same process that wrote the items.

The wording is still authored (by me), which is a smaller and more checkable
problem: `check_leaks` verifies no state label appears in any event, and the
surface-feature baseline in `surface_baseline()` verifies the target cannot be
predicted from option length, position or lexical overlap alone.

CONDITIONS, per anchor
----------------------
  T0  no_evidence   setup only. Correct answer is the OLD state.
  T1  explicit      the new preference is stated outright.
  T2  implicit      the event, from which the new state must be inferred.
  T3  counterfact   SAME old state, DIFFERENT event, DIFFERENT target.
                    Distinguishes computing f(S,E) from memorising E->new.
  T4  conflict      DIFFERENT old state, SAME event, SAME target.
                    Tests whether the answer depends on the old state at all.

Each implicit event gets several paraphrases, so a representation that tracks
the computation can be told apart from one that tracks a token sequence.

TWO SEPARATE LENGTH FACTORS
---------------------------
  n_turns        total conversation turns
  event_pos      where the event sits, as a fraction of the conversation

Varying only n_turns while holding the event at turn 3 confounds length with
distance-to-question. They are generated as independent factors so the two can
be attributed separately.
"""

import argparse, json, random, re, sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Event realizations. Three paraphrases per anchor event, keyed by event id.
# Written to state WHAT HAPPENED and never what the person now wants, and never
# to contain any state label. `check_leaks` enforces the second rule.
# ---------------------------------------------------------------------------
from realizations import R as REALIZATIONS   # keyed "attribute/event_id"


def rkey(attr, ev):
    """Attribute-scoped. `mobility` appeared in two attributes, and a flat id
    key silently gave both the same paraphrases."""
    return f"{attr['attribute']}/{ev['id']}"


NAMES = ["Priya", "Marcus", "Elena", "Tomas", "Aisha", "Ben", "Yuki", "Nadia",
         "Samir", "Clara", "Dmitri", "Rosa", "Iris", "Omar", "Lena", "Hugo"]

FILLER = [
    "Any thoughts on what I should do about the leak in the kitchen?",
    "I need to book a flight next month, no idea when is cheapest.",
    "My neighbour has started practising trumpet at seven in the morning.",
    "I'm trying to decide whether to repaint the hallway.",
    "Do you know anything about repotting a fig tree?",
    "I have to give a short talk at my cousin's wedding.",
    "What is a reasonable amount to spend on a second-hand bicycle?",
    "My laptop fan has started making a grinding noise.",
    "How long can cooked rice stay in the fridge?",
    "I need to write a reference letter for a former colleague.",
    "The council sent a letter about bin collection changes I do not follow.",
    "Is it worth descaling a kettle or just replacing it?",
    "I have a dentist appointment clashing with a meeting.",
    "What is the difference between a will and a living trust?",
    "My tomato plants have yellow leaves at the bottom.",
    "How do I get a wine stain out of a wool rug?",
    "I want to learn a bit of basic carpentry, where do I start?",
    "The pipes make a knocking sound when I turn the hot tap off.",
    "What should I look for when buying a used office chair?",
    "I need to cancel a gym membership and cannot find the terms.",
    "Is there a sensible way to organise twenty years of photographs?",
    "The smoke alarm chirps once an hour and I cannot find which one.",
    "How much notice do I have to give on a rolling tenancy?",
    "My sourdough starter has gone grey on top.",
    "What is a fair price for someone to service a gas boiler?",
    "I keep forgetting to water the plants when I travel.",
    # Added so a 50-turn conversation never repeats a turn: the pool was 26
    # and n_turns=50 reused 22 of them verbatim. Same register as above,
    # and checked to contain no state label of any attribute.
    "The washing machine has started leaving marks on light clothes.",
    "I need to find someone to fix a cracked paving slab out front.",
    "How often should a chimney be swept if it is rarely used?",
    "My passport expires in eight months and I cannot remember the rules.",
    "There is a wasp nest under the eaves at the back of the house.",
    "I have to return a parcel but the label has faded.",
    "What is the best way to get rid of an old mattress?",
    "The bathroom extractor fan has stopped coming on with the light.",
    "I need to set up a standing order and my bank app keeps timing out.",
    "My hedge has grown over the pavement and a neighbour mentioned it.",
    "How do I check whether a tree in the garden has a preservation order?",
    "The front door sticks badly whenever it rains.",
    "I have been asked to be an executor and do not know what it involves.",
    "What paperwork do I need to keep after a car is sold?",
    "My freezer has iced up again three months after defrosting.",
    "I want to put up a shelf but I cannot find the studs.",
    "The recycling bin was not collected and nobody left a note.",
    "How long does it take to get a replacement birth certificate?",
    "There is a damp patch appearing on the ceiling of the spare room.",
    "I need to dispute a parking charge that arrived six weeks late.",
    "What is the safest way to keep paint that has been opened?",
    "My internet drops out every evening at around the same time.",
    "I have to clear out a relative's garage and do not know where to start.",
    "The garden gate hinges have rusted solid over the winter.",
    "How much does it cost to have a window pane replaced?",
    "I keep getting post for someone who moved out years ago.",
    "The radiator upstairs stays cold while the others heat up fine.",
    "What should I check before signing a mobile contract for two years?",
    "My oven thermostat seems to sit hotter than the dial says.",
    "I need advice on repairing a torn lampshade.",
    "The letterbox flap has come loose and bangs in the wind.",
    "How do I find out which company actually supplies my water?",
]

# The setup's MODAL FORCE is the thing under test. "what works for me is X"
# reads as a stated requirement, and the paper's finding is that models treat
# stated requirements as immutable -- so a reviewer can object that the
# instruction-ness was built into the stimulus. The incidental frame reports
# the same state as a habit instead. Everything else is held fixed: same noun,
# same state label, same event, same options.
SETUP_FRAMES = {
    "requirement": "For my {noun}, what works for me is {state}.",
    "incidental": "Lately I have mostly been going with {state} for my "
                  "{noun}.",
}
SETUP = SETUP_FRAMES["requirement"]        # kept for render()/back-compat
EXPLICIT = "What works for me now is {state}, not {old}."
HEADER = ("Below are excerpts from your past conversations with a user, "
          "in chronological order.\n\n")

# ---------------------------------------------------------------------------
# ASK FRAME. How the assistant is asked for its suggestion.
#
# The default writes the last turn as a third-person exam question ("Question:
# Nadia asks you for a suggestion about their way of learning."). That is one
# of six cues marking the item as a benchmark, alongside the excerpt header,
# the "--- Conversation i ---" delimiters, the canned assistant replies, the
# A./B. list and the answer tag. 61.2% of thinking traces mention a test or a
# dataset and some quote this very line back and contrast it with turn 1, so
# whether the framing drives the anchoring is a measurable question and not a
# matter of opinion.
#
#   question     the original. Byte-identical to every result collected so
#                far, so the existing CSVs still join.
#   firstperson  the user asks in their own voice. Header, delimiters, A./B.
#                list and answer tag all held fixed, so this isolates the
#                third-person question line on its own.
#   natural      a real multi-turn chat. No header, no delimiters, no A./B.
#                list and no answer tag, and the transcript is sent as
#                alternating roles rather than pasted into one user message.
#                The two options are named inside the user's last turn so the
#                reply stays scoreable without a tag.
#
# T3 already shows the framing cannot be what blocks the inference, since it
# shares this prompt exactly and is answered correctly about 70% of the time.
# What these arms measure is whether the framing changes the LEVEL of T2.
ASK_FRAMES = {
    "question":    dict(header=True, delims=True, options=True, tag=True,
                        chat=False),
    "firstperson": dict(header=True, delims=True, options=True, tag=True,
                        chat=False),
    "natural":     dict(header=False, delims=False, options=False, tag=False,
                        chat=True),
}
ASK_FIRSTPERSON = "I have to settle on one {noun}. What do you suggest?"
ASK_NATURAL = ("I have to settle on one {noun}. Should I go with {a} or "
               "{b}?")
ANSWER_INSTRUCTION = (
    "\n\nEnd your reply with exactly this line and nothing after it:\n"
    "<answer>OPTION</answer>\nwhere OPTION is exactly one of: {a}, {b}")

CONDITIONS = ["T0_no_evidence", "T1_explicit", "T2_implicit",
              "T3_counterfact", "T4_conflict"]


# ===========================================================================
def load_anchors(path):
    d = json.load(open(path))
    attrs = d["attributes"]
    for a in attrs:
        labels = {s["label"] for s in a["states"]}
        if len(a["states"]) < 3:
            sys.exit(f"{a['attribute']}: needs >=3 states for T3/T4")
        for e in a["events"]:
            for k in ("id", "type", "from", "to", "effect"):
                if k not in e:
                    sys.exit(f"{a['attribute']}/{e.get('id')}: missing {k!r}")
            if e["from"] not in labels or e["to"] not in labels:
                sys.exit(f"{a['attribute']}/{e['id']}: from/to not in states")
            k = f"{a['attribute']}/{e['id']}"
            if k not in REALIZATIONS:
                sys.exit(f"no realizations for {k!r}; add them to realizations.py")
            if len(REALIZATIONS[k]) < 3:
                sys.exit(f"{k}: needs 3 paraphrases, has {len(REALIZATIONS[k])}")
        if a.get("family") not in ("life", "style"):
            sys.exit(f"{a['attribute']}: family must be 'life' or 'style'")
    return attrs



def tokenizer_check(attrs, model_id="Qwen/Qwen3.5-9B", verbose=True):
    """Which state labels are a single token?

    Only matters for the internals subset: logit-lens trajectories and
    next-token probes read one position, so a multi-token label cannot be
    scored there. The behavioural runs parse a tagged answer and are unaffected,
    so multi-token labels stay in the full set.

    Returns {label: n_tokens}. Falls back to a whitespace guess with a loud
    warning if transformers or the network is unavailable.
    """
    labels = sorted({s["label"] for a in attrs for s in a["states"]})
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        n = {l: len(tok.encode(" " + l, add_special_tokens=False)) for l in labels}
        src = model_id
    except Exception as e:
        if verbose:
            print(f"!! could not load {model_id} ({type(e).__name__}); the token")
            print("   counts below are a GUESS. Re-run where the tokenizer is")
            print("   reachable before trusting the internals subset.")
        n = {l: 1 if len(l) <= 6 else 2 for l in labels}
        src = "GUESS"
    if verbose:
        single = [l for l, c in n.items() if c == 1]
        multi = {l: c for l, c in n.items() if c > 1}
        print(f"\nsingle-token check ({src}): {len(single)}/{len(labels)} single")
        if multi:
            print("  multi-token labels (excluded from the internals subset):")
            for l, c in sorted(multi.items(), key=lambda x: -x[1]):
                print(f"    {l:12s} {c} tokens")
    return n


def interp_subset(items, ntok):
    """Items where BOTH displayed options are single tokens."""
    return [it for it in items
            if ntok.get(it["correct"], 9) == 1 and ntok.get(it["distractor"], 9) == 1]


def check_leaks(attrs, verbose=True):
    """No event realization may contain any state label of its attribute."""
    leaks = []
    for a in attrs:
        labels = [s["label"] for s in a["states"]]
        for e in a["events"]:
            for i, txt in enumerate(REALIZATIONS[rkey(a, e)]):
                for lab in labels:
                    if re.search(rf"\b{re.escape(lab.lower())}\b", txt.lower()):
                        leaks.append((a["attribute"], e["id"], i, lab))
    if verbose:
        print("lexical leaks:" if leaks else "no lexical leaks in any realization")
        for l in leaks:
            print("   ", l)
    return leaks


def _counterfactual(attr, ev):
    """Another event from the SAME old state with a DIFFERENT target."""
    for e in attr["events"]:
        if e["from"] == ev["from"] and e["to"] != ev["to"]:
            return e
    return None


def _conflict_state(attr, ev):
    """A third state to use as the old one, keeping the same event and target."""
    for s in attr["states"]:
        if s["label"] not in (ev["from"], ev["to"]):
            return s["label"]
    return None


def build_conversation(item, n_turns, event_pos, rng,
                       ask="question"):
    """setup at turn 1, event at round(event_pos * n_turns), question last."""
    n_turns = max(4, n_turns)
    ev_idx = max(2, min(n_turns - 1, round(event_pos * n_turns)))
    # Over-provision. T0 has no event, so it consumes one MORE filler than the
    # other conditions; sizing this as n_turns-2 made the index wrap and put the
    # same filler in two different turns of the same conversation.
    pool = FILLER[:]
    rng.shuffle(pool)
    need = n_turns
    fills = ([pool[i] for i in range(min(need, len(pool)))]
             + [pool[i % len(pool)] for i in range(len(pool), need)])

    turns, fi = [], 0
    for t in range(1, n_turns + 1):
        if t == 1:
            turns.append((item["setup_text"], "Got it, noted."))
        elif t == ev_idx and item["event_text"] is not None:
            turns.append((item["event_text"], "Understood."))
        else:
            turns.append((fills[fi % len(fills)], "Happy to help with that."))
            fi += 1

    users = [u for u, _ in turns]
    if n_turns <= len(FILLER) and len(set(users)) != len(users):
        raise AssertionError(
            f"repeated turn in a {n_turns}-turn conversation: "
            f"{[u for u in users if users.count(u) > 1][:1]}")

    cfg = ASK_FRAMES[ask]
    n = item["name"]
    if cfg["chat"]:
        # real turns, so the caller can hand them to a chat template
        msgs = []
        for u, a in turns:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        return msgs, ev_idx
    s = HEADER if cfg["header"] else ""
    for i, (u, a) in enumerate(turns, start=1):
        if cfg["delims"]:
            s += f"--- Conversation {i} ---\n{n}: {u}\nAssistant: {a}\n\n"
        else:
            s += f"{n}: {u}\nAssistant: {a}\n\n"
    return s, ev_idx


def build_items(attrs, turns_list, event_positions, n_variants=2, seed=0,
                frames=("requirement",)):
    """One independent pass per setup framing.

    Each frame gets a FRESH rng from the same seed, which buys two things that
    an inner frame loop destroyed: the requirement items come out identical to
    a requirement-only build, so every result already collected still matches
    its item; and each incidental item gets the same name, seed and therefore
    the same filler turns as its requirement twin, so the contrast differs in
    the setup sentence and nothing else.
    """
    out = []
    for frame in frames:
        out.extend(_build_one(attrs, turns_list, event_positions, n_variants,
                              seed, frame))
    return out


def _build_one(attrs, turns_list, event_positions, n_variants, seed, frame):
    rng = random.Random(seed)
    items = []
    for attr in attrs:
        gloss = {s["label"]: s["gloss"] for s in attr["states"]}
        for ev in attr["events"]:
            cf = _counterfactual(attr, ev)
            conflict = _conflict_state(attr, ev)
            for real_i, real in enumerate(REALIZATIONS[rkey(attr, ev)]):
                for cond in CONDITIONS:
                    if cond == "T3_counterfact" and cf is None:
                        continue
                    if cond == "T4_conflict" and conflict is None:
                        continue
                    # T3 shows a DIFFERENT event's text than the anchor it is
                    # indexed by. `shown` is the event whose words the model
                    # actually reads, and its type/source are the ones the
                    # analysis must group on: inheriting the anchor's put the
                    # wrong event_type on 82% of T3 items and the wrong source
                    # on 30% of them, which silently corrupts both the "by
                    # event type" table and the --source author robustness run.
                    shown = ev

                    if cond == "T0_no_evidence":
                        old, new, ev_txt = ev["from"], ev["from"], None
                        distract = ev["to"]
                    elif cond == "T1_explicit":
                        old, new = ev["from"], ev["to"]
                        ev_txt = EXPLICIT.format(state=new, old=old)
                        distract = old
                    elif cond == "T2_implicit":
                        old, new, ev_txt = ev["from"], ev["to"], real
                        distract = old
                    elif cond == "T3_counterfact":
                        old, new = cf["from"], cf["to"]
                        cfr = REALIZATIONS[rkey(attr, cf)]
                        ev_txt = cfr[real_i % len(cfr)]
                        distract = ev["to"]      # the OTHER event's target
                        shown = cf             # <- the event actually displayed
                    else:                        # T4_conflict
                        old, new, ev_txt = conflict, ev["to"], real
                        distract = old

                    for n_turns in turns_list:
                        for epos in event_positions:
                            for v in range(n_variants):
                                # The default frame keeps the ORIGINAL id, so
                                # every result collected so far still joins.
                                sfx = "" if frame == "requirement" else "_f" + frame[:3]
                                items.append(dict(
                                    id=f"{attr['attribute']}_{ev['id']}_r{real_i}"
                                       f"_{cond}_t{n_turns}_p{int(epos*100)}"
                                       f"_v{v}{sfx}",
                                    setup_frame=frame,
                                    attribute=attr["attribute"],
                                    family=attr["family"],
                                    noun=attr["noun"],
                                    source=shown["source"],
                                    origin=shown.get("origin",
                                                     shown["source"]),
                                    event_id=ev["id"],
                                    event_type=shown["type"],
                                    event_force=shown.get("force", ""),
                                    shown_event_id=shown["id"],
                                    anchor_source=ev["source"],
                                    anchor_event_type=ev["type"],
                                    realization=real_i, condition=cond,
                                    n_turns=n_turns, event_pos=epos,
                                    old=old, new=new, distractor=distract,
                                    correct=new,
                                    name=rng.choice(NAMES),
                                    setup_text=SETUP_FRAMES[frame].format(
                                        noun=attr["noun"], state=old),
                                    event_text=ev_txt,
                                    gloss_new=gloss.get(new, ""),
                                    seed=rng.randrange(10**6)))
    return items


def render(item, option_order=0, ask="question"):
    """The stimulus. Returns a string for the two single-message frames and a
    list of chat messages for the `natural` frame."""
    rng = random.Random(item["seed"])
    convo, ev_idx = build_conversation(item, item["n_turns"],
                                       item["event_pos"], rng, ask)
    opts = [item["correct"], item["distractor"]]
    if option_order == 1:
        opts = opts[::-1]
    n = item["name"]
    cfg = ASK_FRAMES[ask]
    if cfg["chat"]:
        ask_text = ASK_NATURAL.format(noun=item["noun"], a=opts[0], b=opts[1])
        body = convo + [{"role": "user", "content": ask_text}]
        return body, opts.index(item["correct"]), opts, ev_idx
    if ask == "firstperson":
        q = f"{n}: " + ASK_FIRSTPERSON.format(noun=item["noun"]) + "\n\n"
    else:
        q = (f"Question: {n} asks you for a suggestion about their "
             f"{item['noun']}. Which should you suggest?\n\n")
    body = convo + q
    if cfg["options"]:
        body += f"A. {opts[0]}\nB. {opts[1]}\n"
    if cfg["tag"]:
        body += ANSWER_INSTRUCTION.format(a=opts[0], b=opts[1])
    # modal_runner.gen_prompt must produce exactly this string; preflight
    # compares them on every condition.
    return body, opts.index(item["correct"]), opts, ev_idx


# ===========================================================================
def surface_baseline(items, n=3000, seed=0):
    """Can the target be predicted with no model at all?

    The Stale-Memory paper found a residual-stream probe at 0.890 balanced
    accuracy and a classifier using only one-hot label codes at 0.884 -- the
    probe barely cleared a baseline that never looked at the model. Run the
    equivalent here at BUILD time: if surface features predict the answer, the
    items are broken before any model sees them.

    TWO baselines, because they answer different questions and only one is a
    gate:

      STRICT      uses only what a bystander could know without reading the
                  conversation -- which two options appear, the attribute, the
                  option lengths, the conversation length and event position.
                  This is the gate. Above 0.60 the items are broken.

      + event_type  adds the hand-annotated 5-way event type. The model NEVER
                  sees this field; obtaining it requires reading the event and
                  classifying it, which is a coarse form of the task itself. It
                  scores far higher (~0.72), and that is expected, not a
                  failure: within an option pair the kind of event is what
                  decides the direction. Reported as a diagnostic so the number
                  is known before a reviewer computes it, and so the claim in
                  the paper is the strict one.

    Sampling is random, not a head slice. items[:600] is one attribute -- with
    three event positions it covers four anchor events, and a classifier fitted
    on that generalises to nothing.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.feature_extraction import DictVectorizer
    except ImportError:
        print("(install scikit-learn to run the surface baseline)")
        return None

    sample = items[:]
    random.Random(seed).shuffle(sample)
    sample = sample[:n]

    def score(use_etype):
        # The label must be the correct VALUE, not the correct POSITION, and
        # the features must be order-invariant. Options are built as [correct,
        # distractor] before the optional swap, so `order` alone would predict
        # position perfectly -- an artifact of the encoding, not of the items.
        X, y = [], []
        for it in sample:
            pair = tuple(sorted([it["correct"], it["distractor"]]))
            f = {
                f"pair={pair[0]}|{pair[1]}": 1,
                f"attr={it['attribute']}": 1,
                "len_sum": len(pair[0]) + len(pair[1]),
                "len_gap": abs(len(pair[0]) - len(pair[1])),
                "n_turns": it["n_turns"],
                "event_pos": it["event_pos"],
            }
            if use_etype:
                f[f"etype={it['event_type']}"] = 1
            X.append(f)
            y.append(it["correct"])
        Xv = DictVectorizer().fit_transform(X)
        yv = np.array(y)
        # rare labels break 5-fold stratification; fold count follows the data
        from collections import Counter as _C
        k = max(2, min(5, min(_C(yv).values())))
        # scale: raw n_turns (4-50) alongside one-hots stops lbfgs converging
        clf = make_pipeline(StandardScaler(with_mean=False),
                            LogisticRegression(max_iter=5000))
        sc = cross_val_score(clf, Xv, yv, cv=k, scoring="balanced_accuracy")
        return float(sc.mean()), float(sc.std()), k

    strict, sd, k = score(False)
    withet, sd2, _ = score(True)
    print(f"\nsurface baseline, n={len(sample)}, {k}-fold, chance 0.500")
    print(f"  STRICT (nothing the model cannot see)  {strict:.3f}  (sd {sd:.3f})")
    print(f"  + event_type (a read of the event)     {withet:.3f}  (sd {sd2:.3f})")
    if strict > 0.60:
        print("  >>> FAIL: knowing which two options appear tells you the answer.")
        print("      Any model accuracy above this is not evidence of inference.")
        print("      Each option pair must be correct in BOTH directions across")
        print("      the set -- add the reverse event, or drop the anchor.")
    else:
        print("  >>> ok: the option pair and the item metadata do not carry the")
        print("      answer. The event_type figure is expected to be high --")
        print("      classifying the event IS a coarse form of the task, and")
        print("      that field is never shown to the model. Quote the strict")
        print("      number; state the other one before a reviewer finds it.")
    return strict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="data/anchors/anchors_template.json")
    ap.add_argument("--turns", default="4,10,20,50")
    ap.add_argument("--positions", default="0.6")
    ap.add_argument("--frames", default="requirement",
                    help="setup framings to generate, comma separated: "
                         + ",".join(SETUP_FRAMES))
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/items_v2.json")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    a = ap.parse_args()

    attrs = load_anchors(a.anchors)
    leaks = check_leaks(attrs)
    assert not leaks, "fix the realizations first"

    turns = [int(x) for x in a.turns.split(",")]
    positions = [float(x) for x in a.positions.split(",")]
    frames = [f.strip() for f in a.frames.split(",") if f.strip()]
    bad = [f for f in frames if f not in SETUP_FRAMES]
    if bad:
        sys.exit(f"unknown --frames {bad}; choose from {list(SETUP_FRAMES)}")
    items = build_items(attrs, turns, positions, a.variants, a.seed,
                        frames=tuple(frames))

    print(f"\n{len(items)} items")
    print("  by condition:", dict(Counter(i["condition"] for i in items)))
    print("  by turns    :", dict(Counter(i["n_turns"] for i in items)))
    print("  by attribute:", dict(Counter(i["attribute"] for i in items)))
    print("  by event type:", dict(Counter(i["event_type"] for i in items)))

    # answer balance: each label must be correct equally often
    bal = defaultdict(Counter)
    for i in items:
        bal[i["attribute"]][i["correct"]] += 1
    print("\n  correct-answer balance per attribute:")
    for k, c in bal.items():
        print(f"    {k:14s} {dict(c)}")

    print("\n  by family   :", dict(Counter(i["family"] for i in items)))
    print("  by source   :", dict(Counter(i["source"] for i in items)))
    print("  by setup frame:", dict(Counter(i["setup_frame"] for i in items)))
    for f in frames:
        ex = next(i for i in items if i["setup_frame"] == f)
        print(f"    {f:12s} {ex['setup_text']!r}")

    surface_baseline(items)

    # ---- internals subset -------------------------------------------------
    ntok = tokenizer_check(attrs, a.tokenizer)
    sub = interp_subset(items, ntok)
    print(f"\ninternals subset: {len(sub)}/{len(items)} items "
          f"({100*len(sub)/max(len(items),1):.1f}%) have single-token options")
    if sub:
        print("  attributes:", dict(Counter(i["attribute"] for i in sub)))
        print("  conditions:", dict(Counter(i["condition"] for i in sub)))
        print("  family    :", dict(Counter(i["family"] for i in sub)))
        sp = a.out.replace(".json", "_interp.json")
        json.dump(sub, open(sp, "w"), indent=1)
        print(f"  wrote {sp}")
        if len(sub) > 50:
            print("\n  surface baseline on the internals subset:")
            surface_baseline(sub)

    if a.inspect:
        for cond in CONDITIONS:
            ex = next((i for i in items if i["condition"] == cond
                       and i["n_turns"] == min(turns)), None)
            if not ex:
                continue
            body, ci, opts, ev_idx = render(ex, 0)
            print(f"\n{'='*70}\n{cond}  [{ex['attribute']}/{ex['event_id']}]  "
                  f"old={ex['old']} -> correct={ex['correct']}  "
                  f"(event at turn {ev_idx})")
            print(body[:1100])

    with open(a.out, "w") as f:
        json.dump(items, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
