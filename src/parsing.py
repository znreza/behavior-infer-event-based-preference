"""Answer extraction from model output. Tested standalone; imported by the runner.

Every failure mode seen in v1 is represented in the test suite at the bottom:
  - Qwen3.5 with thinking on emits NO <think> tags; it writes "Thinking Process:"
    as ordinary prose, so tag-based splitting recovered only 30% of traces
  - the chat template can place the opening <think> in the PROMPT, so the
    generation contains only the CLOSING </think>
  - the answer tag can appear INSIDE the reasoning, where it is a rehearsal and
    not the final answer
  - markdown bold, quotes, trailing punctuation, capitalisation, plurals
  - generation truncated before the tag is emitted
"""
import re

THINK_OPEN = re.compile(r"<think>", re.I)
THINK_CLOSE = re.compile(r"</think>", re.I)
TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
TAG_OPEN_ONLY = re.compile(r"<answer>\s*([^<\n]{1,40})", re.I)
LOOSE = re.compile(r"\banswer\s*[:\-–]\s*\**\s*[\"'`]?([A-Za-z][A-Za-z\- ]{0,30})", re.I)


def split_reasoning(text):
    """-> (reasoning, visible).

    Order matters. A closed <think> block wins; then a bare closing tag, which
    is the normal Qwen case when the template opens the block in the prompt;
    then an unclosed opening tag, which means generation ran out mid-trace. If
    there are no tags at all, everything before the final <answer> is treated as
    reasoning -- models reason in plain prose more often than they tag it.
    """
    t = str(text)
    o, c = THINK_OPEN.search(t), THINK_CLOSE.search(t)
    if o and c and c.start() > o.start():
        return t[o.end():c.start()].strip(), (t[:o.start()] + t[c.end():]).strip()
    if c:                                   # opened in the prompt
        return t[:c.start()].strip(), t[c.end():].strip()
    if o:                                   # opened, never closed
        return t[o.end():].strip(), t[:o.start()].strip()
    tags = list(TAG.finditer(t)) or list(re.finditer(r"<answer>", t, re.I))
    if tags:
        i = tags[-1].start()
        return t[:i].strip(), t[i:].strip()
    return "", t.strip()


def _norm(w):
    w = str(w).strip().lower()
    w = w.strip("*_`\"'“”‘’ \t\n.,;:!?)(][}{")
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
    for i, x in enumerate(o):                # plurals, truncation, "X option"
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
    """-> (answer or None, method). `opts` are the two displayed option strings."""
    reasoning, visible = split_reasoning(text)

    # 0. the tag was echoed unfilled; the answer is on the preceding line
    hit = _placeholder_echo(text, opts)
    if hit:
        return hit, "placeholder_echo"

    # 1. closed tag in the visible section -- the requested format
    for src, name in ((visible, "tag"), (str(text), "tag_anywhere")):
        m = TAG.findall(src)
        if m:
            hit = _match(m[-1], opts)
            if hit:
                return hit, name
            hit = _letter_ref(m[-1], opts)
            if hit:
                return hit, name + "_letter"

    # 2. an opening tag with no close: truncated mid-answer
    m = TAG_OPEN_ONLY.findall(visible) or TAG_OPEN_ONLY.findall(str(text))
    if m:
        hit = _match(m[-1], opts)
        if hit:
            return hit, "tag_unclosed"
        hit = _letter_ref(m[-1], opts)
        if hit:
            return hit, "tag_unclosed_letter"

    # 3. "Answer: X" without tags
    m = LOOSE.findall(visible) or LOOSE.findall(str(text))
    if m:
        hit = _match(m[-1], opts)
        if hit:
            return hit, "loose"
        hit = _letter_ref(m[-1], opts)
        if hit:
            return hit, "loose_letter"

    # 4. the visible section names exactly one option
    body = visible or str(text)
    for line in [l for l in body.strip().split("\n") if l.strip()][::-1]:
        hits = {o for o in opts
                if re.search(rf"\b{re.escape(o.lower())}\b", line.lower())}
        if len(hits) == 1:
            return hits.pop(), "last_line"

    # 5. last resort: last option mentioned anywhere in the visible section
    pos = {o: body.lower().rfind(o.lower()) for o in opts}
    pos = {o: p for o, p in pos.items() if p >= 0}
    if len(pos) == 1:
        return next(iter(pos)), "sole_mention"
    return None, "unparsed"
