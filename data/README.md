# Data

What is in git, and what is not.

| Path | In git | Notes |
|---|---|---|
| `anchors/` | yes | the 77 authored causal claims, and the templates they were written in |
| `annotation/` | yes | the completed annotation, the blank form, the instructions, the answer key |
| `items_v2.json` | no | 27,720 generated items, 22 MB |
| `items_v2_interp.json` | no | 26,064 of those whose options are single tokens, 21 MB |

The two item files are on Hugging Face at HF_URL, or rebuild them from the
anchors, which are in git:

```bash
python src/generate_v2.py --anchors data/anchors/anchors_full.json \
    --turns 4,10,20,50 --out data/items_v2.json
```

The rebuild is deterministic and `src/preflight.py` asserts that the result is
byte-identical to the released file. The anchors are the source of truth; the
items are derived from them and never the other way round.

## Item schema

One JSON list. Each element is one item.

| Field | Meaning |
|---|---|
| `id` | unique, encodes attribute, event, condition, turns, position and wording |
| `condition` | `T0_no_evidence`, `T1_explicit`, `T2_implicit`, `T3_counterfact`, `T4_conflict` |
| `attribute` | which preference is at stake, one of 12 |
| `name` | the user's name in this rendering |
| `setup_text` | turn 1, where the user states a preference |
| `setup_frame` | `requirement` ("what works for me is X") or `incidental` ("lately I have been going with X") |
| `event_text` | the evidence turn; empty for T0 |
| `event_id` | which of the 77 authored events this is |
| `event_type` | the authored category of the event |
| `event_force` | `pull` if the event names a replacement, `push` if it only rules the stated option out |
| `realization` | which of three wordings of the event is used, 0 to 2 |
| `old` | the option stated in turn 1 |
| `new` | the option the event makes workable |
| `correct` | the keyed answer, always equal to `new` |
| `distractor` | the other option offered |
| `gloss_new` | a short readable gloss of `new` |
| `noun` | the noun phrase the question is asked about |
| `n_turns` | conversation length: 4, 10, 20 or 50 |
| `event_pos` | how deep the event sits: 0.3, 0.6 or 0.9 |
| `family` | the group of mutually exclusive options this attribute draws from |
| `source` | provenance of the causal claim; uniform `author`, every anchor is author-validated |
| `origin` | whether the event was first drafted by the author or by a model, before validation |
| `anchor_source`, `anchor_event_type` | the same fields on the anchor the item came from |
| `shown_event_id` | T3 only: the counterfactual event actually shown, which always differs from the anchor |
| `seed` | the RNG seed for this item's filler selection |

`correct` equals `new` in every condition including T4, where the keyed answer
follows the event and the preference stated in turn 1 is the distractor.

Read T4 on the `pull` subset only. On `push` items the event names no
replacement, so keeping the stated value is defensible and a wrong answer is
not evidence of anchoring.

## Design

The conditions are crossed with four factors, and every
`(condition, n_turns, event_pos)` cell holds 462 items:

- 5 conditions
- 4 conversation lengths: 4, 10, 20, 50
- 3 event depths: 0.3, 0.6, 0.9
- 2 turn-1 phrasings: `requirement`, `incidental`
- 3 wordings per event

T0 and T2 share an option pair with opposite keyed answers, so a model that
copies the setup scores 100% on T0 and 0% on T2. That pairing is what makes
the T0-to-T2 difference readable.

A logistic classifier on surface features (option pair, attribute, text
lengths, conversation length, event position) reaches 0.508 balanced accuracy
against a 0.500 baseline over 3,000 sampled items under five-fold
cross-validation, so the answer is not recoverable without reading the event.

## Annotation

| File | What it is |
|---|---|
| `annotator_A_filled.csv` | the completed annotation, 60 items |
| `annotator_A_form.csv` | the blank form it was filled from |
| `INSTRUCTIONS.txt` | what the annotator was told, verbatim |
| `_answer_key.json` | the keys, withheld from the annotator |

Alongside each answer the annotator recorded two judgements about the item,
made independently of the answer given:

- `old_ok` — would it still be appropriate to suggest the original option?
- `best_pick` — of the two offered, is the answer given clearly the better one?

These distinguish an annotator who rejects an item's premise from one who
fails to draw the inference. Gating on both gives the paper's cleanest
comparison. Reproduce it with:

```bash
python src/analyze_human.py
```

Both fields are free text, so a value may be capitalised. Lowercase before
matching; a case-sensitive comparison silently drops a row and was the cause
of one wrong number in an earlier draft.

One annotator answered the set, so there is no inter-annotator agreement and
the premise judgements are that reader's own. The figures bound the human
ceiling rather than estimating it.
