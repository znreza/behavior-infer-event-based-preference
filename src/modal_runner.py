#!/usr/bin/env python3
"""
Runs the diagnostic on Modal: the forced-choice probe, free-text
generation, and every ask framing. Consumes data/items_v2.json from
generate_v2.py and writes one CSV plus a .meta.json per run to results/.

    # 0. which prompt format puts the options in contention? ALWAYS FIRST
    modal run modal_runner.py --experiment probecheck --models "Qwen/Qwen3.5-9B"

    # 1. forced-choice logit probe (fast, non-deliberative)
    modal run modal_runner.py --experiment probe --fmt chat_prefill \
        --conditions T0,T1,T2 --turns 4 \
        --models "Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B"

    # 2. generation, reasoning captured
    modal run modal_runner.py --experiment generate --thinking \
        --conditions T0,T1,T2,T3,T4 --turns 4 --max-items 600 \
        --models "Qwen/Qwen3.5-9B"

    # 3. same model, reasoning off
    modal run modal_runner.py --experiment generate --no-thinking ... 

    # 4. turn ablation
    modal run modal_runner.py --experiment probe --turns 4,10,20,50 ...

WHY EACH GUARD EXISTS
---------------------
probecheck   a raw completion prompt makes an instruct model emit "\\n\\n" first,
             so the two option letters hold almost no probability and the A-vs-B
             comparison is decided by whatever the model wanted to say instead.
ab_mass      recorded per row so that gate can be checked after the fact too.
per order    a model that always answers "A" scores exactly 50% once orders are
             counterbalanced, which is indistinguishable from guessing unless
             the two orders are reported separately.
parse method fallback parsers differed from the strict tag by 20-30 points in
             v1; pooling them mixes two populations.
truncated    a reply cut off before the tag is not a wrong answer.
"""

import os
import re
import json
import random
import modal

TRANSFORMERS = "transformers>=5.15,<6"     # Qwen3.5 needs >= Feb 2026
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.0", TRANSFORMERS, "accelerate",
                 "numpy>=2.0", "sentencepiece", "protobuf")
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("preference-v2")
GPU = "A100-40GB"
MIN = 60

# ===========================================================================
# RENDERING -- must match generate_v2.py exactly
# ===========================================================================
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
HEADER = ("Below are excerpts from your past conversations with a user, "
          "in chronological order.\n\n")
ANSWER_INSTRUCTION = (
    "\n\nEnd your reply with exactly this line and nothing after it:\n"
    "<answer>OPTION</answer>\nwhere OPTION is exactly one of: {a}, {b}")
# ASK FRAME. Must stay identical to generate_v2.ASK_FRAMES; preflight compares
# the rendered stimulus on every frame. See the long note in generate_v2.py
# for why the framing is a factor rather than an argument.
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
# Gemma 3 has no thinking mode and no enable_thinking flag, so deliberation
# has to be elicited by the prompt. This also makes the comparison fairer
# across families: it tests deliberation itself rather than one vendor's
# implementation of it.
# --cot 1 (DIRECTED) names the event and asks what it implies. That is not
# neutral deliberation -- it hands the model the task structure, so a gain
# under it cannot be attributed to thinking. --cot 2 (NEUTRAL) asks only for
# step-by-step reasoning, which is what Qwen's native thinking mode provides.
# Run both to separate deliberation from task-direction.
COT_INSTRUCTIONS = {
    1: ("\n\nBefore answering, work through the conversation step by step: "
        "what the person said they preferred, what has happened since, and "
        "what that implies for what you should suggest now."),
    2: "\n\nThink step by step before answering.",
}
COT_INSTRUCTION = COT_INSTRUCTIONS[1]      # back-compat


def build_conversation(item, ask="question"):
    rng = random.Random(item["seed"])
    n_turns = max(4, item["n_turns"])
    ev_idx = max(2, min(n_turns - 1, round(item["event_pos"] * n_turns)))
    pool = FILLER[:]
    rng.shuffle(pool)
    fills = ([pool[i] for i in range(min(n_turns, len(pool)))]
             + [pool[i % len(pool)] for i in range(len(pool), n_turns)])

    turns, fi = [], 0
    for t in range(1, n_turns + 1):
        if t == 1:
            turns.append((item["setup_text"], "Got it, noted."))
        elif t == ev_idx and item["event_text"] is not None:
            turns.append((item["event_text"], "Understood."))
        else:
            turns.append((fills[fi % len(fills)], "Happy to help with that."))
            fi += 1
    cfg = ASK_FRAMES[ask]
    if cfg["chat"]:
        msgs = []
        for u, a in turns:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        return msgs, ev_idx
    s = HEADER if cfg["header"] else ""
    for i, (u, a) in enumerate(turns, start=1):
        if cfg["delims"]:
            s += (f"--- Conversation {i} ---\n{item['name']}: {u}\n"
                  f"Assistant: {a}\n\n")
        else:
            s += f"{item['name']}: {u}\nAssistant: {a}\n\n"
    return s, ev_idx


def question_body(item, option_order, ask="question"):
    """Conversation + question + the A./B. option lines.

    The option lines belong HERE, not in probe_prompt. They used to be added
    only on the probe path, so the probe and the generation arm showed the model
    two different stimuli and "two methods agreeing on the same items" compared
    two different prompts. This matches generate_v2.render() exactly; preflight
    asserts that it still does.
    """
    convo, ev_idx = build_conversation(item, ask)
    opts = [item["correct"], item["distractor"]]
    if option_order == 1:
        opts = opts[::-1]
    cfg = ASK_FRAMES[ask]
    if cfg["chat"]:
        ask_text = ASK_NATURAL.format(noun=item["noun"], a=opts[0], b=opts[1])
        return convo + [{"role": "user", "content": ask_text}], opts, ev_idx
    if ask == "firstperson":
        q = (f"{item['name']}: "
             + ASK_FIRSTPERSON.format(noun=item["noun"]) + "\n\n")
    else:
        q = (f"Question: {item['name']} asks you for a suggestion about their "
             f"{item['noun']}. Which should you suggest?\n\n")
    if cfg["options"]:
        q += f"A. {opts[0]}\nB. {opts[1]}\n"
    return convo + q, opts, ev_idx


def probe_prompt(item, option_order, tok, fmt="chat_prefill"):
    body, opts, _ = question_body(item, option_order)
    body += "\nAnswer with a single letter. The answer is option:"
    if fmt == "raw":
        return body, opts
    if fmt == "raw_nl":
        return body + "\n\n", opts
    msgs = [{"role": "user", "content": body}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return body + "\n\n", opts
    if fmt == "chat_prefill":
        text += "The answer is option:"
    return text, opts


def gen_prompt(item, option_order, tok, thinking, cot=0, ask="question"):
    body, opts, _ = question_body(item, option_order, ask)
    cfg = ASK_FRAMES[ask]
    if cfg["chat"]:
        # already a turn list. A CoT instruction is appended to the user's
        # last turn so the arm still reads as one person speaking.
        msgs = [dict(m) for m in body]
        if cot:
            msgs[-1]["content"] += COT_INSTRUCTIONS[int(cot)]
    else:
        if cot:
            body += COT_INSTRUCTIONS[int(cot)]
        if cfg["tag"]:
            body += ANSWER_INSTRUCTION.format(a=opts[0], b=opts[1])
        msgs = [{"role": "user", "content": body}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(msgs, enable_thinking=thinking, **kw), opts
    except TypeError:
        return tok.apply_chat_template(msgs, **kw), opts


# ===========================================================================
# PARSING -- see parsing.py for the standalone test suite (15/15)
# ===========================================================================
THINK_OPEN = re.compile(r"<think>", re.I)
THINK_CLOSE = re.compile(r"</think>", re.I)
TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
TAG_OPEN_ONLY = re.compile(r"<answer>\s*([^<\n]{1,40})", re.I)
LOOSE = re.compile(r"\banswer\s*[:\-–]\s*\**\s*[\"'`]?([A-Za-z][A-Za-z\- ]{0,30})", re.I)


def split_reasoning(text):
    t = str(text)
    o, c = THINK_OPEN.search(t), THINK_CLOSE.search(t)
    if o and c and c.start() > o.start():
        return t[o.end():c.start()].strip(), (t[:o.start()] + t[c.end():]).strip()
    if c:
        return t[:c.start()].strip(), t[c.end():].strip()
    if o:
        return t[o.end():].strip(), t[:o.start()].strip()
    tags = list(TAG.finditer(t)) or list(re.finditer(r"<answer>", t, re.I))
    if tags:
        i = tags[-1].start()
        return t[:i].strip(), t[i:].strip()
    return "", t.strip()


def _norm(w):
    w = str(w).strip().lower().strip("*_`\"'“”‘’ \t\n.,;:!?)(][}{")
    return re.sub(r"\s+", " ", w)


def _match(word, opts):
    w = _norm(word)
    o = [_norm(x) for x in opts]
    if w in o:
        return opts[o.index(w)]
    # Require 3 characters before allowing a prefix match. Five state labels
    # begin with "a" or "b" (ambient, batch, books, brief, bullets), so a bare
    # letter answer used to prefix-match one of them and silently return the
    # wrong option -- "b" resolved to "batch" regardless of which option "B"
    # actually was. Below 3 chars, only an exact or letter match is safe.
    for i, x in enumerate(o):
        if len(w) >= 3 and (w.startswith(x) or x.startswith(w)
                            or w.endswith(" " + x)):
            return opts[i]
    # An option word sitting inside a phrase: "the driving option", "option B,
    # transit". Without this the tag content fails to match and extract_answer
    # drops through to `last_line` -- which in v1 was the low-accuracy parse
    # population (63.8% vs 84.1% for `tag`), so a correctly tagged reply got
    # counted in the bucket used to argue the fallbacks are unreliable.
    # Only when EXACTLY ONE option appears; two means genuinely ambiguous.
    hits = [opts[i] for i, x in enumerate(o)
            if x and re.search(rf"\b{re.escape(x)}\b", w)]
    if len(hits) == 1:
        return hits[0]
    return None


LETTER_REF = re.compile(r"^(?:option\s*)?([ab])$", re.I)


def _letter_ref(word, opts):
    """Resolve a letter answer: "A" -> the option displayed as A.

    Both arms now show "A. x / B. y", so a model can answer with the letter
    instead of the value even where the instruction asks for the value. `opts`
    is in DISPLAY order, so A is opts[0] and B is opts[1] under either
    counterbalancing. Recorded as its own parse method rather than folded into
    `tag`, because a letter answer is a different behaviour from naming the
    value and should be visible in the parse-method breakdown.
    """
    m = LETTER_REF.match(_norm(word))
    return opts["ab".index(m.group(1).lower())] if m else None

PLACEHOLDER = re.compile(r"^\(?\s*option\s*\)?$", re.I)


def _placeholder_echo(text, opts):
    """The model copied the instruction's placeholder instead of filling it in.

        Cycling
        <answer>OPTION</answer>

    "End your reply with exactly this line" is taken literally, and the real
    answer sits on the line BEFORE the tag. split_reasoning cuts at the last
    <answer>, so that line goes into `reasoning` and only the placeholder is
    left in `visible` -- the answer is present and was being thrown away.
    Qwen2.5-7B-Instruct did this on 55% of its replies, unevenly by condition
    (23% parsed on T0 against 72% on T3), so it silently reshaped the sample.
    """
    t = str(text)
    tags = TAG.findall(t)
    if not tags or not all(PLACEHOLDER.match(_norm(x) or "x") for x in tags):
        return None
    m = list(TAG.finditer(t))[-1]
    # The instruction says "End your reply with exactly this line", so this
    # only applies when the tag really is at the end. A reasoning model also
    # restates the constraint while planning -- at a median 5% into the reply
    # with 8000 characters still to come -- and there the preceding line is
    # mid-deliberation, not an answer. Reading it as one scored 28 truncated
    # T2 rows at 42.9%, near chance, and lifted the condition from 5% to 9%.
    # Strip special tokens in BOTH conventions before measuring what follows.
    # The <|...|> form is Qwen's; Gemma uses <end_of_turn> and <pad>, so a
    # Qwen-only regex left 128 characters of padding after the tag and this
    # check declined on 82 Gemma-4B replies that were plain placeholder echoes.
    after = re.sub(r"<\|[^>]*\|>|</?(?:pad|eos|bos|s|end_of_turn|"
                   r"start_of_turn|endoftext|im_end|im_start)>", "",
                   t[m.end():]).strip()
    if len(after) > 40:
        return None
    pre = t[:m.start()]
    for line in [l for l in pre.strip().split("\n") if l.strip()][::-1]:
        hit = _match(line, opts) or _letter_ref(line, opts)
        if hit:
            return hit
        seen = {o for o in opts
                if re.search(rf"\b{re.escape(o.lower())}\b", line.lower())}
        if len(seen) == 1:
            return seen.pop()
    return None


def extract_answer_natural(text, opts):
    """Score a reply that was never asked for a tag.

    The `natural` ask frame removes the answer tag, so there is nothing to
    parse and the reply has to be read. The rule is fixed here BEFORE the run
    so it cannot be tuned afterwards to whichever reading gives a nicer
    number:

      1. If the model volunteers a tag anyway, honour it. That is an answer,
         not an artefact.
      2. Otherwise take the option whose LAST mention sits latest in the
         reply, matched on word boundaries. A recommendation reply names the
         thing it recommends last, and this survives an opening "you asked
         about books or videos".
      3. A reply naming neither option is unparsed. It is counted and
         reported, never dropped.

    The returned flag `both_in_tail` marks replies where the two options are
    mentioned within 120 characters of each other at the end, which is where
    rule 2 is least safe ("rather than videos, go with books"). Those rows are
    scored but reported separately so the arm can be read with and without
    them.
    """
    t = str(text)
    _, visible = split_reasoning(t)
    for src in (visible, t):
        m = TAG.findall(src)
        if m:
            hit = _match(m[-1], opts) or _letter_ref(m[-1], opts)
            if hit:
                return hit, "tag_volunteered", False
    last = {}
    for o in opts:
        found = [mm.start() for mm in
                 re.finditer(rf"\b{re.escape(o.lower())}\b", visible.lower())]
        if found:
            last[o] = found[-1]
    if not last:
        return None, "no_option_named", False
    pick = max(last, key=last.get)
    both_in_tail = (len(last) == 2
                    and abs(last[opts[0]] - last[opts[1]]) <= 120)
    return pick, "last_mention", both_in_tail


def extract_answer(text, opts):
    reasoning, visible = split_reasoning(text)
    hit = _placeholder_echo(text, opts)
    if hit:
        return hit, "placeholder_echo"
    for src, name in ((visible, "tag"), (str(text), "tag_anywhere")):
        m = TAG.findall(src)
        if m:
            hit = _match(m[-1], opts)
            if hit:
                return hit, name
            hit = _letter_ref(m[-1], opts)
            if hit:
                return hit, name + "_letter"
    m = TAG_OPEN_ONLY.findall(visible) or TAG_OPEN_ONLY.findall(str(text))
    if m:
        hit = _match(m[-1], opts)
        if hit:
            return hit, "tag_unclosed"
        hit = _letter_ref(m[-1], opts)
        if hit:
            return hit, "tag_unclosed_letter"
    m = LOOSE.findall(visible) or LOOSE.findall(str(text))
    if m:
        hit = _match(m[-1], opts)
        if hit:
            return hit, "loose"
        hit = _letter_ref(m[-1], opts)
        if hit:
            return hit, "loose_letter"
    body = visible or str(text)
    for line in [l for l in body.strip().split("\n") if l.strip()][::-1]:
        hits = {o for o in opts
                if re.search(rf"\b{re.escape(o.lower())}\b", line.lower())}
        if len(hits) == 1:
            return hits.pop(), "last_line"
    pos = {o: body.lower().rfind(o.lower()) for o in opts}
    pos = {o: p for o, p in pos.items() if p >= 0}
    if len(pos) == 1:
        return next(iter(pos)), "sole_mention"
    return None, "unparsed"


# ===========================================================================
def _clip(t, keep=0, head=1500):
    """Store the text WHOLE by default. keep=0 means no clipping at all.

    Set keep>0 only on an explicit --raw-chars, and it keeps head + TAIL
    rather than head only.

    `raw=text[:4000]` looked harmless and silently destroyed the record: the
    answer tag is at the END of a reply, so a 6500-char reasoning trace stored
    head-first loses exactly the part needed to re-score it. 100% of the
    reasoning-on replies were clipped this way. The runner parses the full text
    before storing, so its own numbers were right -- but the CSV could not be
    re-parsed after a parser fix, which is most of the value of keeping raw.
    """
    t = str(t)
    if keep <= 0 or len(t) <= keep:
        return t
    return (t[:head] + f"\n...[{len(t) - keep} chars omitted]...\n"
            + t[-(keep - head):])


def _dtype_for(model_id):
    """bfloat16 for Gemma, float16 otherwise.

    Gemma was trained in bfloat16 and its activations overflow float16's range
    -- loading it in fp16 yields inf/NaN logits and silently useless numbers
    rather than an error. Qwen is fine in fp16 and it is faster, so this is
    per-family rather than global.
    """
    import torch
    return torch.bfloat16 if "gemma" in model_id.lower() else torch.float16


def _load_model(model_id):
    import torch
    import transformers as tf
    from transformers import AutoTokenizer
    token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(model_id, token=token,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=_dtype_for(model_id), device_map="cuda", token=token)

    # Ask the config which class this checkpoint wants, instead of guessing.
    # Blind fallback made AutoModelForCausalLM load 49% of gemma-3-27b before
    # failing, and the partial allocation was never freed -- so the correct
    # loader then OOM'd on a card that had room for it.
    from transformers import AutoConfig
    order = []
    try:
        arch = (getattr(AutoConfig.from_pretrained(model_id, token=token),
                        "architectures", None) or [None])[0]
        if arch and getattr(tf, arch, None):
            order.append(arch)
    except Exception:
        arch = None
    order += [n for n in ("AutoModelForCausalLM",
                          "Gemma3ForConditionalGeneration",
                          "AutoModelForImageTextToText") if n not in order]

    errs = []
    for name in order:
        cls = getattr(tf, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **kw).eval()
            print(f"[{model_id}] loaded via {name}, dtype={kw['dtype']}"
                  + (f" (config said {arch})" if arch else ""), flush=True)
            return tok, model
        except Exception as e:
            errs.append(f"{name}: {type(e).__name__}: {str(e)[:160]}")
            # free whatever the failed attempt put on the card before the
            # next one tries
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("no loader worked:\n  " + "\n  ".join(errs))


def _letter_ids(tok):
    ids = []
    for L in "AB":
        c = [tok.encode(" " + L, add_special_tokens=False),
             tok.encode(L, add_special_tokens=False)]
        ids.append(next((x[0] for x in c if len(x) == 1), c[0][0]))
    return ids


_LOGIT_KW = None


def _last_logits(model, enc):
    """Logits for the final position only.

    A full forward returns [batch, seq, vocab]. At batch 16 and 1700 tokens
    (the 50-turn cell) that tensor alone is 8.3 GB in fp16, on top of 18.5 GB
    of weights for the 9B -- close enough to 40 GB to fail. Only the last
    position is ever read, so ask for just that one where the API allows it.
    The keyword was renamed between transformers versions; probe once, cache.
    """
    global _LOGIT_KW
    if _LOGIT_KW is None:
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                out = model(**enc, **{kw: 1})
                _LOGIT_KW = kw
                print(f"  (using {kw}=1: last-position logits only)")
                return out.logits[:, -1, :]
            except TypeError:
                continue
        _LOGIT_KW = ""
        print("  (this transformers build wants the full logits tensor; "
              "lower --batch-size if the 50-turn cell OOMs)")
    if _LOGIT_KW:
        return model(**enc, **{_LOGIT_KW: 1}).logits[:, -1, :]
    return model(**enc).logits[:, -1, :]


SEC = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})]
VOL = {"/root/.cache/huggingface": hf_cache}


@app.function(image=image, gpu=GPU, volumes=VOL, timeout=30 * MIN, secrets=SEC)
def run_probecheck(model_id: str, items: list):
    import torch
    tok, model = _load_model(model_id)
    hf_cache.commit()
    lids = _letter_ids(tok)
    out = []
    with torch.no_grad():
        for fmt in ("raw", "raw_nl", "chat", "chat_prefill"):
            mass = []
            for it in items[:40]:
                p, opts = probe_prompt(it, 0, tok, fmt)
                enc = tok(p, return_tensors="pt").to(model.device)
                pr = torch.softmax(model(**enc).logits[0, -1].float(), -1)
                top = int(pr.argmax())
                mass.append(float(pr[lids].sum()))
                out.append(dict(model=model_id, fmt=fmt,
                                ab_mass=mass[-1], top_token=tok.decode([top]),
                                top_is_letter=bool(top in lids)))
            print(f"[{model_id}] {fmt:14s} median P(A)+P(B) = "
                  f"{sorted(mass)[len(mass)//2]:.3f}")
    return out


@app.function(image=image, gpu=GPU, volumes=VOL, timeout=120 * MIN, secrets=SEC)
def run_probe(model_id: str, items: list, fmt: str = "chat_prefill",
              batch_size: int = 16):
    import torch
    try:
        tok, model = _load_model(model_id)
    except Exception as e:
        print(f"[{model_id}] LOAD FAILED: {e}")
        return [dict(model=model_id, id="LOAD_FAILED", error=str(e)[:300])]
    hf_cache.commit()
    lids = _letter_ids(tok)
    jobs = [(it, o) + probe_prompt(it, o, tok, fmt)
            for it in items for o in (0, 1)]
    # Sort by prompt length before batching. data/items_v2.json interleaves turn
    # counts, so an unsorted batch in the turn ablation mixes 200-token and
    # 1700-token prompts: every batch then pays the 50-turn memory and pad
    # cost. Sorting is analysis-neutral -- each row carries its own metadata.
    jobs.sort(key=lambda j: len(j[2]))
    print(f"[{model_id}] {len(jobs)} forward passes, fmt={fmt}, bs={batch_size}, "
          f"prompt chars {len(jobs[0][2])}..{len(jobs[-1][2])}")

    rows = []
    with torch.no_grad():
        for s in range(0, len(jobs), batch_size):
            chunk = jobs[s:s + batch_size]
            enc = tok([j[2] for j in chunk], return_tensors="pt",
                      padding=True).to(model.device)
            lg = _last_logits(model, enc)
            if s == 0 and not torch.isfinite(lg).all():
                raise RuntimeError(
                    f"{model_id}: non-finite logits on the first batch. "
                    "Almost always a dtype problem -- Gemma overflows in "
                    "float16. Check _dtype_for().")
            pr = torch.softmax(lg.float(), dim=-1)
            for k, (it, order, _, opts) in enumerate(chunk):
                v = torch.tensor([lg[k, i].item() for i in lids])
                pred = int(v.argmax())
                top = int(pr[k].argmax())
                # Store the probability on the CORRECT option and the margin,
                # not just which side won. Without them calibration and
                # confidence-weighted accuracy need a whole new run -- the same
                # mistake as storing text[:4000]: a field cheap to keep and
                # impossible to reconstruct. v1 recorded p_correct; v2 dropped
                # it.
                p2 = torch.softmax(v.float(), dim=-1)
                ci = 0 if opts[0] == it["correct"] else 1
                pc = float(p2[ci])
                margin = float(abs(v[0] - v[1]))
                rows.append(dict(
                    model=model_id, fmt=fmt, id=it["id"],
                    attribute=it["attribute"], family=it["family"],
                    event_id=it["event_id"],
                    condition=it["condition"], event_type=it["event_type"],
                    n_turns=it["n_turns"], event_pos=it["event_pos"],
                    event_force=it.get("event_force", ""),
                    source=it["source"], origin=it.get("origin", ""),
                    realization=it["realization"], order=order,
                    opt_a=opts[0], opt_b=opts[1],
                    picked=opts[pred], correct=it["correct"],
                    is_correct=bool(opts[pred] == it["correct"]),
                    p_correct=pc, logit_margin=margin,
                    p_a=float(p2[0]), p_b=float(p2[1]),
                    ab_mass=float(pr[k, lids].sum()),
                    top_token=tok.decode([top]),
                    top_is_letter=bool(top in lids)))
            if s % (batch_size * 10) == 0:
                print(f"[{model_id}] {s}/{len(jobs)}", flush=True)
    return rows


@app.function(image=image, gpu=GPU, volumes=VOL, timeout=480 * MIN, secrets=SEC)
def run_generate(model_id: str, items: list, thinking: bool = True,
                 max_new_tokens: int = 2048, batch_size: int = 16,
                 sample: int = -1, gen_seed: int = 0,
                 force_answer: int = 0, raw_chars: int = 0,
                 cot: int = 0, ask: str = "question"):
    """sample: -1 = decide from `thinking`, 0 = greedy, 1 = sample.

    Greedy decoding in thinking mode is the wrong default and it cost us a run.
    Qwen's own guidance for reasoning mode is temperature 0.6 / top_p 0.95 /
    top_k 20, and it warns that greedy decoding produces endless repetition. A
    looping sequence never emits EOS, and generate() runs until the LONGEST
    sequence in the batch finishes -- so a single loop drags all 16 to the token
    cap. Measured effect on the first phase-3 attempt: >3 min per batch of 8,
    projecting ~7-8 h against a 130-min estimate and a 480-min timeout.
    Sampling with a fixed seed is both faster and what the model card asks for.

    force_answer=1 turns on budget forcing. Any sequence still running at the
    cap has "</think>\n\n<answer>" appended and gets ~16 more tokens, so it
    has to commit. Without it a truncated trace carries no tag, and the weak
    fallbacks then pick whichever option was named last in an INCOMPLETE
    deliberation -- in v1 those rows scored 63.8% against 84.1% for a real
    tag, i.e. close to noise dressed as an answer. Forcing converts missing
    data into a declared intervention; every forced row is flagged so the
    headline can be reported with and without them.
    """
    import torch
    try:
        tok, model = _load_model(model_id)
    except Exception as e:
        print(f"[{model_id}] LOAD FAILED: {e}")
        return [dict(model=model_id, id="LOAD_FAILED", raw=str(e)[:300])]
    hf_cache.commit()

    # Does enable_thinking actually DO anything? A Jinja chat template silently
    # ignores unknown kwargs, so `except TypeError` is not a test: Gemma 3
    # accepts enable_thinking=True and renders exactly the same prompt. Trusting
    # that would have tagged a Gemma run "thinkon" with thinking=True recorded
    # while nothing changed. Compare the rendered strings instead.
    _m = [{"role": "user", "content": "x"}]
    _kw = dict(tokenize=False, add_generation_prompt=True)
    base = tok.apply_chat_template(_m, **_kw)
    try:
        on = tok.apply_chat_template(_m, enable_thinking=True, **_kw)
        off = tok.apply_chat_template(_m, enable_thinking=False, **_kw)
        has_think = (on != base) or (off != base) or (on != off)
    except TypeError:
        has_think = False
    use_think = thinking and has_think
    do_sample = bool(use_think) if sample < 0 else bool(sample)
    # Qwen's documented thinking-mode sampling settings. The plan passes
    # --sample 1 on BOTH generation arms so the reasoning flag stays the only
    # difference between them; --sample 0 forces greedy if you want it.
    gen_kw = dict(do_sample=do_sample)
    if do_sample:
        gen_kw.update(temperature=0.6, top_p=0.95, top_k=20)
        torch.manual_seed(gen_seed)
    print(f"[{model_id}] thinking supported={has_think} used={use_think} "
          f"cot={int(cot)}"
          + ({0: "", 1: " (DIRECTED -- names the event)",
              2: " (neutral)"}[int(cot)]))
    if thinking and not has_think:
        print(f"[{model_id}] !! --thinking was requested but this template has "
              "no enable_thinking. The run is NOT a reasoning run; use --cot 1 "
              "to elicit deliberation by prompt instead.", flush=True)
    print(f"[{model_id}] decoding: "
          + ("greedy" if not do_sample else
             f"sample T=0.6 top_p=0.95 top_k=20 seed={gen_seed}"), flush=True)

    jobs = [(it, o) + gen_prompt(it, o, tok, use_think, int(cot), ask)
            for it in items for o in (0, 1)]
    jobs.sort(key=lambda j: len(j[2]))
    nb = (len(jobs) + batch_size - 1) // batch_size
    print(f"[{model_id}] {len(jobs)} generations in {nb} batches, "
          f"cap {max_new_tokens}, bs={batch_size}", flush=True)

    import time as _t
    rows, done_tok, t0 = [], 0, _t.time()
    n_cap_total = n_seq_total = 0
    with torch.no_grad():
        for s in range(0, len(jobs), batch_size):
            chunk = jobs[s:s + batch_size]
            enc = tok([j[2] for j in chunk], return_tensors="pt",
                      padding=True).to(model.device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id, **gen_kw)
            L = enc["input_ids"].shape[1]
            gen = tok.batch_decode(out[:, L:], skip_special_tokens=False)

            # ---- budget forcing on the sequences still running at the cap --
            forced = {}
            per_len = (out[:, L:] != tok.pad_token_id).sum(dim=1)
            stuck = [i for i in range(len(chunk))
                     if int(per_len[i]) >= max_new_tokens - 2]
            if force_answer and stuck:
                suf = ("\n</think>\n\n<answer>" if use_think
                       else "\n\n<answer>")
                enc2 = tok([chunk[i][2] + gen[i] + suf for i in stuck],
                           return_tensors="pt", padding=True).to(model.device)
                out2 = model.generate(**enc2, max_new_tokens=16,
                                      do_sample=False,
                                      pad_token_id=tok.pad_token_id)
                L2 = enc2["input_ids"].shape[1]
                cont = tok.batch_decode(out2[:, L2:], skip_special_tokens=True)
                for j, i in enumerate(stuck):
                    forced[i] = cont[j]

            for k, (it, order, _, opts) in enumerate(chunk):
                text = gen[k]
                reasoning, visible = split_reasoning(text)
                if ask == "natural":
                    ans, how, both_tail = extract_answer_natural(text, opts)
                else:
                    ans, how = extract_answer(text, opts)
                    both_tail = False
                was_forced = False
                if k in forced:
                    # score the forced continuation as if it were the tag it
                    # was made to produce, and only if the free-running parse
                    # found nothing trustworthy
                    fa, fh = extract_answer("<answer>" + forced[k] + "</answer>",
                                            opts)
                    if (ask != "natural" and fa is not None
                            and how in ("unparsed", "last_line",
                                        "sole_mention")):
                        ans, how, was_forced = fa, fh + "_forced", True
                ntok = int((out[k, L:] != tok.pad_token_id).sum())
                rows.append(dict(
                    model=model_id, id=it["id"], attribute=it["attribute"],
                    family=it["family"], event_id=it["event_id"],
                    condition=it["condition"],
                    event_type=it["event_type"], n_turns=it["n_turns"],
                    event_pos=it["event_pos"],
                    event_force=it.get("event_force", ""),
                    source=it["source"], origin=it.get("origin", ""),
                    realization=it["realization"],
                    order=order, opt_a=opts[0], opt_b=opts[1],
                    thinking=use_think, thinking_supported=has_think,
                    cot_prompt=int(cot),
                    ask_frame=ask,
                    both_options_in_tail=both_tail,
                    setup_frame=it.get("setup_frame", "requirement"),
                    # Evidence the flag actually changed the decoding. Do NOT
                    # use n_reason_chars for this: split_reasoning treats
                    # everything before <answer> as reasoning, so it is > 0 in
                    # 99.4% of reasoning-OFF rows too (measured on the v1 CSVs).
                    # Qwen3.5 opens <think> in the PROMPT and writes "Thinking
                    # Process:" as prose, so the closing tag is the only tag
                    # that ever reaches the generation.
                    has_think_close=bool(THINK_CLOSE.search(text)),
                    has_think_prose=bool(
                        re.search(r"thinking process", text, re.I)),
                    parsed=ans is not None, parse_method=how, picked=ans,
                    correct=it["correct"],
                    is_correct=bool(ans == it["correct"]),
                    picked_old=bool(ans == it["old"]),
                    n_gen_tokens=ntok,
                    n_reason_chars=len(reasoning),
                    forced=was_forced,
                    forced_text=forced.get(k, "") or "",
                    truncated=bool(ntok >= max_new_tokens - 2),
                    # was `.split()[:8]` -- only the first eight words of the
                    # event counted, so an event whose distinctive wording came
                    # later scored as unmentioned. All content words now.
                    reason_has_event=bool(
                        reasoning and it.get("event_text") and any(
                            w in reasoning.lower()
                            for w in str(it["event_text"]).lower().split()
                            if len(w) > 5)),
                    reason_has_old=bool(reasoning and re.search(
                        rf"\b{re.escape(it['old'].lower())}\b", reasoning.lower())),
                    reason_has_new=bool(reasoning and re.search(
                        rf"\b{re.escape(it['new'].lower())}\b", reasoning.lower())),
                    n_raw_chars=len(str(text)),
                    raw_clipped=bool(raw_chars > 0
                                     and len(str(text)) > raw_chars),
                    reasoning=_clip(reasoning, raw_chars, raw_chars // 4),
                    raw=_clip(text, raw_chars, raw_chars // 4)))
            bi = s // batch_size + 1
            gen_tok = int((out[:, L:] != tok.pad_token_id).sum())
            done_tok += gen_tok
            el = _t.time() - t0
            eta = (nb - bi) * el / bi / 60
            # Count the SEQUENCES at the cap, not whether the batch reached
            # it. The boolean form flagged a whole batch of 16 whenever one
            # row failed to stop, which is ~97% of batches at a per-row rate
            # of only 0.20 -- it looked catastrophic and carried no number.
            per = (out[:, L:] != tok.pad_token_id).sum(dim=1)
            ncap = int((per >= max_new_tokens - 2).sum())
            n_cap_total += ncap
            n_seq_total += len(chunk)
            print(f"[{model_id}] batch {bi}/{nb}  {s + len(chunk)}/{len(jobs)}  "
                  f"{gen_tok/max(len(chunk),1):.0f} tok/seq  "
                  f"{done_tok/max(el,1e-9):.0f} tok/s  "
                  f"elapsed {el/60:.1f}m  eta {eta:.0f}m  "
                  f"at-cap {ncap}/{len(chunk)} "
                  f"(run {100*n_cap_total/max(n_seq_total,1):.0f}%)", flush=True)
    return rows


# ===========================================================================
@app.local_entrypoint()
def main(experiment: str = "probe",
         models: str = "Qwen/Qwen3.5-9B",
         items_file: str = "data/items_v2.json",
         conditions: str = "", turns: str = "", positions: str = "",
         frames: str = "", attributes: str = "", family: str = "",
         source: str = "",
         realization: int = -1,
         max_items: int = 0, seed: int = 0,
         fmt: str = "chat_prefill", thinking: bool = True,
         max_new_tokens: int = 2048, batch_size: int = 0,
         sample: int = -1, gen_seed: int = 0, force_answer: int = 0,
         raw_chars: int = 0, cot: int = 0, ask: str = "question",
         gpu: str = "",
         out: str = "results_v2", tag: str = "",
         shards: int = 1, stratify: bool = True):
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit(
            "pandas is missing from the LOCAL environment. main() is a Modal\n"
            "local_entrypoint, so it runs on this machine, not in the image:\n"
            "    pip install 'modal>=0.64' pandas")

    if ask not in ASK_FRAMES:
        raise SystemExit(f"unknown --ask {ask!r}; "
                         f"choose from {list(ASK_FRAMES)}")
    if ask != "question" and experiment.startswith("probe"):
        # the probe appends "Answer with a single letter. The answer is
        # option:", which is itself an exam cue, so a non-default ask frame on
        # the probe path would measure a mixture of the two.
        raise SystemExit("--ask only applies to the generate experiment; the "
                         "probe prompt carries its own answer instruction")
    if ask == "natural" and force_answer:
        raise SystemExit("--force-answer needs an answer tag to force and "
                         "--ask natural removes it")

    items = json.load(open(items_file))
    n0 = len(items)
    if conditions:
        want = {c.strip() for c in conditions.split(",")}
        items = [i for i in items
                 if i["condition"] in want or i["condition"].split("_")[0] in want]
    if turns:
        want = {int(t) for t in turns.split(",")}
        items = [i for i in items if i["n_turns"] in want]
    if positions:
        # event_pos is a separate length factor from n_turns. Without this
        # filter every run pooled all three positions, so a run meant to hold
        # the reference cell fixed silently varied distance-to-question too.
        want = {float(x) for x in positions.split(",")}
        items = [i for i in items if float(i["event_pos"]) in want]
    if frames:
        want = {f.strip() for f in frames.split(",")}
        items = [i for i in items
                 if i.get("setup_frame", "requirement") in want]
    if attributes:
        want = {a.strip() for a in attributes.split(",")}
        items = [i for i in items if i["attribute"] in want]
    if family:
        items = [i for i in items if i["family"] == family]
    if source:
        items = [i for i in items if i["source"] == source]
    if realization >= 0:
        items = [i for i in items if i["realization"] == realization]
    if max_items and len(items) > max_items:
        rng = random.Random(seed)
        if stratify:
            # Plain shuffle-and-truncate left the conditions unbalanced (127 vs
            # 113 at max_items=600). Take an equal share per condition so the
            # per-condition n in the paper's table is not an artifact of the
            # subsample.
            from collections import defaultdict as _dd
            byc = _dd(list)
            for i in items:
                byc[i["condition"]].append(i)
            per = max_items // len(byc)
            picked = []
            for c in sorted(byc):
                g = byc[c][:]
                rng.shuffle(g)
                picked.extend(g[:per])
            rng.shuffle(picked)
            items = picked
        else:
            rng.shuffle(items)
            items = items[:max_items]
    print(f"{n0} items in {items_file} -> {len(items)} after filters")
    if not items:
        raise SystemExit("no items match the filters")

    from collections import Counter
    print("  conditions:", dict(Counter(i["condition"] for i in items)))
    print("  turns     :", dict(Counter(i["n_turns"] for i in items)))
    print("  event_pos :", dict(Counter(i["event_pos"] for i in items)))
    print("  frame     :", dict(Counter(i.get("setup_frame", "requirement")
                                        for i in items)))

    os.makedirs(out, exist_ok=True)
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    def _with_gpu(fn):
        """Apply --gpu. GPU is bound at decoration time, so a 27B model needs
        an override -- and this has to happen for EVERY experiment. It used to
        live below the probecheck branch, which returns early, so --gpu was
        silently ignored there and a 54GB model was sent to a 40GB card."""
        if not gpu:
            return fn
        try:
            out = fn.with_options(gpu=gpu)
            print(f"GPU override: {gpu}")
            return out
        except Exception as e:
            raise SystemExit(
                f"--gpu {gpu} not applicable ({type(e).__name__}). Edit the "
                f"module-level GPU constant (currently {GPU}) instead.")

    if experiment == "probecheck":
        rows = []
        pc_fn = _with_gpu(run_probecheck)
        for r in pc_fn.starmap([(m, items) for m in model_list]):
            rows.extend(r)
        df = pd.DataFrame(rows)
        # probecheck used the bare name, so it had neither the --tag nor the
        # overwrite guard the other two experiments got: a second phase-0 run
        # silently replaced the first.
        pc = f"{out}/probecheck__{tag}.csv" if tag else f"{out}/probecheck.csv"
        if os.path.exists(pc):
            raise SystemExit(
                f"{pc} already exists -- refusing to overwrite a completed "
                "run. Pass a distinct --tag or move the file.")
        df.to_csv(pc, index=False)
        print(f"wrote {pc}  ({len(df)} rows)")
        with open(pc.replace(".csv", ".meta.json"), "w") as f:
            json.dump(dict(experiment="probecheck", models=model_list,
                           items_file=items_file, conditions=conditions,
                           turns=turns, positions=positions,
                           max_items=max_items, seed=seed, tag=tag,
                           n_items=len(items), n_rows=len(df)), f, indent=1)
        for m in model_list:
            d = df[df.model == m]
            if not len(d):
                continue
            print(f"\n=== {m} ===")
            t = d.groupby("fmt").agg(mass=("ab_mass", "median"),
                                     letter=("top_is_letter", "mean"))
            t["letter"] = (t["letter"] * 100).round(1)
            print(t.round(3).to_string())
            best = t.mass.idxmax()
            print(f"  best: {best!r} (mass {t.mass.max():.3f})")
            if t.mass.max() < 0.5:
                print("  >>> NO format works. Use --experiment generate instead;")
                print("      a one-token forced choice is not measurable here.")
        return

    if experiment not in ("probe", "generate"):
        raise SystemExit(f"unknown --experiment {experiment!r}; "
                         "expected probecheck, probe or generate")
    # starmap fans out one container per argument tuple. Fanning out over
    # MODELS alone means a two-model run gets two containers no matter how
    # many items it carries, which is why a 4620-generation run sat on two
    # GPUs for over an hour. --shards splits the items as well, so the
    # container count is len(models) * shards.
    #
    # Assignment is round-robin over the filtered list, not contiguous
    # slices: the list is grouped by condition, so contiguous slices would
    # hand one container every T0 item and another every T4 item, and the
    # containers would finish at very different times. Round-robin keeps
    # every shard balanced across conditions, turn counts and orders.
    #
    # The cost is one model load per container. That is a fixed ~1-3 min, so
    # sharding pays off on long runs and is waste on short ones.
    if shards < 1:
        raise SystemExit("--shards must be at least 1")
    if shards > len(items):
        raise SystemExit(f"--shards {shards} exceeds {len(items)} items")
    chunks = [items[i::shards] for i in range(shards)]
    sizes = sorted({len(c) for c in chunks})
    print(f"sharding {len(items)} items into {shards} x ~{sizes[-1]} "
          f"across {len(model_list) * shards} containers")
    if experiment == "probe":
        fn = run_probe
        args = [(m, c, fmt, batch_size or 16)
                for m in model_list for c in chunks]
    else:
        fn = run_generate
        args = [(m, c, thinking, max_new_tokens, batch_size or 16,
                 sample, gen_seed, force_answer, raw_chars, cot, ask)
                for m in model_list for c in chunks]
    fn = _with_gpu(fn)
    rows = []
    for r in fn.starmap(args):
        rows.extend(r)
    df = pd.DataFrame(rows)
    if experiment == "probe":
        base = f"{experiment}_{fmt}"
    else:
        mode = "thinkon" if thinking else "thinkoff"
        if cot:
            mode += {1: "_cotdir", 2: "_cotneu"}[int(cot)]
        base = f"generate_{mode}"
    # A --tag is required in practice: phases 1, 2 and 4 are all
    # `--experiment probe --fmt chat_prefill`, so without it the third run
    # silently overwrites the first two.
    name = f"{base}__{tag}" if tag else base
    path = f"{out}/{name}.csv"
    if os.path.exists(path):
        raise SystemExit(
            f"{path} already exists -- refusing to overwrite a completed run.\n"
            "Pass a distinct --tag (e.g. --tag phase2_turns) or move the file.")
    df.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(df)} rows)")
    meta = dict(experiment=experiment, models=model_list, items_file=items_file,
                conditions=conditions, turns=turns, positions=positions,
                frames=frames, cot=cot, gpu=gpu or GPU, attributes=attributes,
                family=family, source=source, realization=realization,
                max_items=max_items, seed=seed,
                stratify=stratify, fmt=fmt, thinking=thinking,
                max_new_tokens=max_new_tokens, batch_size=batch_size,
                sample=sample, gen_seed=gen_seed,
                force_answer=force_answer, raw_chars=raw_chars,
                ask=ask, shards=shards,
                tag=tag, n_items=len(items), n_rows=len(df))
    with open(path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    failed = df[df.get("id", "") == "LOAD_FAILED"] if "id" in df else df.iloc[:0]
    if len(failed):
        print("!! failed to load:", list(failed.model.unique()))
        df = df[df.id != "LOAD_FAILED"]
    for m in model_list:
        d = df[df.model == m]
        if len(d):
            summarize(d, m, experiment)


def _wilson(k, n, z=1.96):
    """Wilson score interval. A run can sit at 0/40 or 40/40 in a cell and the
    normal approximation gives a zero-width interval there."""
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def _paired_t1_t2(d):
    """Sign test on T1 vs T2 over the same anchor, paraphrase, order and length.

    The headline is a within-anchor contrast: same causal claim, same option
    pair, same option order -- the event stated outright versus implied. The
    filler turns and the user's name differ between the two (the generator
    draws them per condition), so this is a matched pair, not the identical
    prompt. Comparing two marginal accuracies discards the pairing entirely.
    """
    if "event_id" not in d:
        return None
    t1, t2 = {}, {}
    for r in d.itertuples():
        k = (r.attribute, r.event_id, r.realization, r.order, r.n_turns)
        if r.condition == "T1_explicit":
            t1[k] = bool(r.is_correct)
        elif r.condition == "T2_implicit":
            t2[k] = bool(r.is_correct)
    both = set(t1) & set(t2)
    if not both:
        return None
    win = sum(1 for k in both if t1[k] and not t2[k])
    los = sum(1 for k in both if t2[k] and not t1[k])
    n = win + los
    if n == 0:
        return dict(n_pairs=len(both), t1_only=0, t2_only=0, p=1.0)
    from math import comb
    lo = min(win, los)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / (2 ** n))
    return dict(n_pairs=len(both), t1_only=win, t2_only=los, p=p)


def summarize(d, model_id, experiment):
    import numpy as np
    from collections import Counter
    print(f"\n{'='*74}\n{model_id}  [{experiment}]\n{'='*74}")

    if experiment == "probe":
        print("\n--- validity: is the forced choice measuring a decision? ---")
        t = d.groupby("condition").agg(mass=("ab_mass", "median"),
                                       letter=("top_is_letter", "mean"))
        t["letter"] = (t["letter"] * 100).round(1)
        print(t.round(3).to_string())
        if d.ab_mass.median() < 0.5:
            print("  >>> the options hold under half the probability mass. The")
            print("      accuracies below are not interpretable. Most common")
            print("      intended token:", Counter(d.top_token).most_common(3))
    else:
        print("\n--- output format ---")
        print("   ", dict(Counter(d.parse_method).most_common()))
        print(f"    parsed {100*d.parsed.mean():.1f}%   "
              f"truncated {100*d.truncated.mean():.1f}%")
        if d.truncated.mean() > 0.05:
            print(f"    >>> {100*d.truncated.mean():.1f}% hit the token cap. In v1"
                  " this was 20.8% with a 900-token")
            print("        cap and reasoning on. Raise --max-new-tokens and re-run;")
            print("        a reply cut off before the tag is not a wrong answer.")
        # Pooled truncation hides the thing that actually biases the result:
        # if T2 makes the model deliberate longer, T2 truncates more, and
        # analysing only parsed rows leaves T2 weighted toward its easy items.
        print("\n--- deliberation length and truncation, by condition ---")
        print(f"    {'condition':16s} {'n':>5s} {'med tok':>8s} {'p90':>6s} "
              f"{'trunc %':>8s} {'parsed %':>9s}")
        for c in sorted(d.condition.unique()):
            g = d[d.condition == c]
            print(f"    {c:16s} {len(g):5d} "
                  f"{g.n_gen_tokens.median():8.0f} "
                  f"{g.n_gen_tokens.quantile(.9):6.0f} "
                  f"{100*g.truncated.mean():8.1f} "
                  f"{100*g.parsed.mean():9.1f}")
        if "forced" in d and d.forced.any():
            fo = d[d.forced]
            print(f"\n--- budget forcing: {len(fo)} rows "
                  f"({100*len(fo)/len(d):.1f}%) were made to commit ---")
            print(f"    accuracy among forced rows   "
                  f"{100*fo[fo.parsed].is_correct.mean():5.1f}%")
            nf = d[~d.forced & d.parsed]
            print(f"    accuracy among the rest      "
                  f"{100*nf.is_correct.mean():5.1f}%")
            print("    Report the headline both ways. If they agree, forcing is")
            print("    bookkeeping; if they diverge, say which you quote and why.")

        if d.truncated.any() and (~d.truncated).any():
            print("\n--- truncated vs complete replies ---")
            for lab, g in (("complete", d[~d.truncated]),
                           ("truncated", d[d.truncated])):
                pk = g[g.parsed]
                acc = f"{100*pk.is_correct.mean():5.1f}" if len(pk) else "    -"
                print(f"    {lab:10s} n={len(g):5d}  parsed {100*g.parsed.mean():5.1f}%"
                      f"  acc among parsed {acc}%")
            print("    A truncated reply is not a wrong answer. If the two rows")
            print("    differ in accuracy, dropping unparsed rows is not neutral.")

        tr = d.groupby("condition").truncated.mean() * 100
        if len(tr) > 1 and tr.max() - tr.min() > 10:
            print(f"    >>> truncation ranges {tr.min():.1f}-{tr.max():.1f}% "
                  f"across conditions ({tr.idxmax()} worst).")
            print("        Differential truncation: dropping unparsed rows will")
            print("        bias the conditions unequally. Either raise the cap")
            print("        or report accuracy over ALL rows with truncated")
            print("        replies counted separately, not silently dropped.")

        if "has_think_close" in d:
            print("\n--- did the thinking flag take effect? ---")
            print(f"    requested={bool(d.thinking.iloc[0])}   "
                  f"</think> in output {100*d.has_think_close.mean():.1f}%   "
                  f"'Thinking Process' {100*d.has_think_prose.mean():.1f}%")
            print(f"    n_gen_tokens median {d.n_gen_tokens.median():.0f} "
                  f"(v1: 57 off vs 838 on -- the real discriminator)")
        if d.parse_method.nunique() > 1:
            t = (d[d.parsed].groupby(["condition", "parse_method"])
                 .is_correct.agg(["mean", "size"]))
            t["mean"] = (t["mean"] * 100).round(1)
            print("\n--- accuracy by parse method (fallbacks are not neutral) ---")
            print(t.to_string())
        d = d[d.parsed]

    print("\n--- accuracy by condition, with option order split ---")
    print(f"{'condition':16s} {'n':>5s} {'acc %':>7s} {'95% CI':>15s} "
          f"{'order0':>7s} {'order1':>7s} {'min':>7s}  verdict")
    for c in sorted(d.condition.unique()):
        s = d[d.condition == c]
        o0 = s[s.order == 0].is_correct.mean() * 100
        o1 = s[s.order == 1].is_correct.mean() * 100
        lo, hi = _wilson(int(s.is_correct.sum()), len(s))
        acc = 100 * s.is_correct.mean()
        mn = min(o0, o1)
        # A 40-point gap was too lax: T3 ran at 86.6 vs 46.8 (39.8 points) and
        # did not trip it, while its min-order sat below chance -- i.e. the
        # pooled number was position bias, not ability. Judge on the min order
        # against chance, which is what the printed advice always said.
        if hi < 50:
            v = "SYSTEMATIC ERROR"      # confidently wrong, not guessing
        elif mn > 55 and lo > 50:
            v = "ability"
        elif abs(o0 - o1) > 25:
            v = "ORDER-DRIVEN"          # pooled is not ability
        else:
            v = "at chance"
        ci = f"[{lo:.1f},{hi:.1f}]"
        print(f"{c:16s} {len(s):5d} {acc:7.1f} {ci:>15s} "
              f"{o0:7.1f} {o1:7.1f} {mn:7.1f}  {v}")
    print("  chance = 50%. `ability` needs BOTH orders above chance and the CI")
    print("  clear of 50. SYSTEMATIC ERROR means the whole interval is BELOW")
    print("  50: the model is confidently wrong, which is a result, not noise.")
    print("  ORDER-DRIVEN means the pooled number is position bias; quote the")
    print("  min-order column instead.")

    # Where the old value is a displayed option, the wrong answer IS the stale
    # preference, so accuracy alone hides the mechanism.
    if "opt_a" in d and "opt_b" in d:
        print("\n--- what the model picked instead ---")
        stale = d[d.picked.notna()]
        # `correct` is the post-event state; in T0/T1/T2/T4 the distractor is
        # the value named in the setup, so not-correct == kept the stale value.
        t = 100 * (1 - stale.groupby("condition").is_correct.mean())
        for c, v in t.items():
            note = ("  <- setup value is not an option here"
                    if c.startswith("T3") else "  <- kept the setup value")
            print(f"    {c:16s} {v:5.1f}%{note}")

    if "p_correct" in d:
        print("\n--- confidence, not just which side won ---")
        print(f"{'condition':16s} {'acc %':>7s} {'mean p(correct)':>16s} "
              f"{'median':>7s} {'p>0.5':>7s} {'near-tie':>9s} {'med margin':>11s}")
        for c in sorted(d.condition.unique()):
            g = d[d.condition == c]
            tie = 100 * (g.logit_margin < 0.5).mean()
            print(f"{c:16s} {100*g.is_correct.mean():7.1f} "
                  f"{g.p_correct.mean():16.3f} {g.p_correct.median():7.3f} "
                  f"{100*(g.p_correct > .5).mean():6.1f}% {tie:8.1f}% "
                  f"{g.logit_margin.median():11.2f}")
        print("  p(correct) is the softmax over the two option letters, so 0.5")
        print("  is indifference. A cell at 3% accuracy AND p(correct)~0.05 has")
        print("  ruled the right answer out; one at 3% with p(correct)~0.45 is")
        print("  losing coin flips. `near-tie` = logit margin under 0.5.")
        # order-driven cells: is it near-ties resolved by position, or
        # confident position bias? Different claims about the same number.
        for c in sorted(d.condition.unique()):
            g = d[d.condition == c]
            o0 = g[g.order == 0].is_correct.mean() * 100
            o1 = g[g.order == 1].is_correct.mean() * 100
            if abs(o0 - o1) <= 25:
                continue
            pa = g.p_a.mean()
            print(f"  {c}: order gap {abs(o0-o1):.0f} pts, mean p(A) = {pa:.3f}, "
                  f"median margin {g.logit_margin.median():.2f}")
            print(f"     -> {'near-ties broken by position' if g.logit_margin.median() < 1 else 'CONFIDENT preference for the A slot'}")

    pt = _paired_t1_t2(d)
    if pt:
        print(f"\n--- T1 vs T2, paired within anchor/realization/order ---")
        print(f"    {pt['n_pairs']} pairs   T1-only-correct {pt['t1_only']}   "
              f"T2-only-correct {pt['t2_only']}   sign test p = {pt['p']:.2e}")
        if pt["p"] > 0.05:
            print("    >>> the explicit-vs-implicit gap is not significant here.")

    if d.n_turns.nunique() > 1:
        print("\n--- by turn count (length; event_pos held fixed) ---")
        print((d.pivot_table(index="condition", columns="n_turns",
                             values="is_correct") * 100).round(1).to_string())
    if "event_pos" in d and d.event_pos.nunique() > 1:
        print("\n--- by event position (distance to question; length fixed) ---")
        print((d.pivot_table(index="condition", columns="event_pos",
                             values="is_correct") * 100).round(1).to_string())
        print("  event_pos x n_turns gives the event's turn index; a decline "
              "here\n  with n_turns fixed is distance, not length.")
    if "event_force" in d and d.event_force.notna().any():
        t4 = d[d.condition == "T4_conflict"]
        if len(t4) and t4.event_force.nunique() > 1:
            print("\n--- T4 split on event force (do NOT pool these) ---")
            for f, g in t4.groupby("event_force"):
                o0 = g[g.order == 0].is_correct.mean() * 100
                o1 = g[g.order == 1].is_correct.mean() * 100
                lo, hi = _wilson(int(g.is_correct.sum()), len(g))
                acc = 100 * g.is_correct.mean()
                if f == "pull":
                    note = ("keyed answer IS derivable -> read as accuracy")
                else:
                    note = ("keyed answer is NOT derivable; the event is about "
                            "a state\n                      the user no longer "
                            "uses, so keeping the setup value is\n              "
                            "        correct and this number is INVERTED")
                print(f"    {f:5s} n={len(g):4d}  acc {acc:5.1f} "
                      f"[{lo:.1f},{hi:.1f}]  order {o0:.1f}/{o1:.1f}")
                print(f"          -> {note}")
                if f == "push":
                    print(f"          -> correct-as-'keep the stated value': "
                          f"{100 - acc:.1f}%")

    print("\n--- by family ---")
    print((d.pivot_table(index="condition", columns="family",
                         values="is_correct") * 100).round(1).to_string())
    print("\n--- by event type ---")
    print((d.pivot_table(index="condition", columns="event_type",
                         values="is_correct") * 100).round(1).to_string())
    if "origin" in d and d.origin.nunique() > 1:
        print("\n--- by who first drafted the anchor (all author-validated) ---")
        print((d.pivot_table(index="condition", columns="origin",
                             values="is_correct") * 100).round(1).to_string())
        print("  Diagnostic only -- every claim was endorsed by the author, so")
        print("  do not report this as a robustness split.")

    if experiment != "probe" and "reason_has_new" in d and d.n_reason_chars.max() > 0:
        ig = d[d.condition == "T2_implicit"]
        if len(ig):
            print(f"\n--- reasoning traces on T2, n={len(ig)} ---")
            print(f"    median chars           {ig.n_reason_chars.median():.0f}")
            print(f"    mentions the event     {100*ig.reason_has_event.mean():.1f}%")
            print(f"    mentions the OLD value {100*ig.reason_has_old.mean():.1f}%")
            print(f"    mentions the NEW value {100*ig.reason_has_new.mean():.1f}%")
            both = ig[ig.reason_has_old & ig.reason_has_new]
            if len(both):
                print(f"    mentions BOTH          {100*len(both)/len(ig):.1f}%, "
                      f"and answers correctly {100*both.is_correct.mean():.1f}%")
