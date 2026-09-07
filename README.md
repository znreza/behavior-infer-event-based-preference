# Why Personalized LLM Agents Fail at Implicit Preference Updates

A user tells an assistant what they prefer. Later, something happens to them
that makes that preference unusable, and they never say so directly. Does the
assistant notice?

This repository holds the diagnostic, the evaluation code and the analysis
behind the paper. Across nine models, assistants follow a preference change
the user **states** on close to 100% of items and infer the same change from
an **event** on 2.2% to 13.3% of them, below a random baseline. A human
annotator resolves the same items at 93.3%.

The failure has two directions, and the interaction sets the balance between
them. Under an examination-style prompt the preference stated first carries
the force of an instruction, which produces near-perfect compliance when the
change is stated and near-total anchoring when it must be inferred. Recasting
the same exchange as ordinary chat improves the inference and makes the
assistant start overriding instructions the user has just given. No framing
we test avoids both.

- **Paper:** ARXIV_URL
- **Dataset:** HF_URL
- **Contact:** zarreen.reza@joulesai.com

## The diagnostic

Each item is a simulated conversation ending in a two-option choice. Turn 1
states a preference; one later turn carries the evidence; the remaining turns
are filler drawn from a pool of 58. The five conditions differ **only** in
that one evidence turn, and all conditions hold the same length, so a
difference between them cannot come from length, topic or option position.

| | The evidence turn contains | Keyed answer |
|---|---|---|
| **T0** | a filler question; nothing changes | the option stated in turn 1 |
| **T1** | the user stating the change in words | the newly stated option |
| **T2** | an event that implies the change without naming the new option | the option the event makes workable |
| **T3** | a different event implying a third option, with the stated preference withheld from the choices | the option that event implies |
| **T4** | the T2 event, but turn 1 states an unrelated preference | the option the event implies |

T2 is the condition the paper turns on. T3 is the control that shows the
inference is available: take the stated preference off the ballot and accuracy
jumps from single digits to roughly 60–75%.

The conditions are crossed with conversation length (4, 10, 20, 50 turns),
event depth (30%, 60%, 90%), turn-1 phrasing (a requirement or a casual
habit) and three wordings per event, giving 27,720 items over 12 preference
attributes and 77 authored events.

## Getting the data

The generated items and the per-item model outputs are large and live on
Hugging Face rather than in git. Everything else is here.

```bash
pip install -r requirements.txt

# Option A: download the released data (recommended).
hf download HF_REPO --repo-type dataset --local-dir hf
mkdir -p data results
cp hf/items/items.json               data/items_v2.json
cp hf/items/items_single_token.json  data/items_v2_interp.json
cp hf/model_outputs/*/*.csv          results/
cp hf/model_outputs/*/*.meta.json    results/

# Option B: rebuild the items from the authored anchors, which are in git.
python src/generate_v2.py --anchors data/anchors/anchors_full.json \
    --turns 4,10,20,50 --out data/items_v2.json
```

Option B reproduces `items_v2.json` byte for byte; `src/preflight.py` asserts
this. See [`data/README.md`](data/README.md) for the item schema.

## Reproducing the results

Every figure and every table in the paper is recomputed from the released
result files. No GPU is needed.

```bash
python src/figs/plots.py                 # the four figures, into figures/
python src/analyze_human.py              # the human ceiling and the gated subsets
python src/check_fillers.py              # the filler-adjacency control
python src/analyze.py results/<file>.csv # summarise any single run
python src/figs/trace_patterns.py        # how the traces justify a wrong answer
```

Every number is recomputed from the stored CSVs and nothing is copied between
outputs, so a disagreement between two of them is a real bug rather than a
transcription slip.

The related-work comparison is reproducible too, and needs network access to
load the third-party datasets:

```bash
python src/check_hb_events.py            # the HorizonBench overlap measurement
python src/check_realpref.py             # the RealPref overlap measurement
python src/check_dataset.py              # what each benchmark records, and where
```

## Re-running the models

Inference runs on [Modal](https://modal.com). Two gates come first, and both
exit non-zero if the run should not start.

```bash
python src/test_parsing.py               # parser cases, plus a drift check
                                         # against the runner's inlined copy
python src/preflight.py                  # data, design, leakage, rendering,
                                         # resources, retention, ground truth
export HF_TOKEN=...                      # the Gemma repos are gated
bash src/run_all.sh                      # the full plan, phase by phase
```

`preflight.py` is worth running before any GPU spend. It checks that the items
are byte-identical to a fresh build from the anchors, that every design cell
is balanced, that no option label leaks into any event text, that the runner
renders what the generator says it renders, and that each run retains the
columns needed to answer the question it was launched for. It reports blockers
separately from judgement calls.

`run_all.sh` drives everything through `src/modal_runner.py`. The runner
inlines its own copy of the answer parser, because the Modal image does not
carry local source; `test_parsing.py` fails if the two copies drift.

Scoring free-text replies under the natural framing uses two blinded judges
that never see the keyed answer:

```bash
python src/judge_traces.py   ...   # does the trace derive the consequence?
python src/judge_choice.py   ...   # which option did the reply recommend?
```

## Layout

```
data/
  anchors/            the 77 authored causal claims and the authoring templates
  annotation/         the completed annotation, the blank form, the withheld
                      key, and the instructions given to the annotator
  items_v2.json       the generated items, fetched or rebuilt (see above)
results/              per-item model outputs, including reasoning traces,
                      fetched from the dataset (see above)
figures/              written by src/figs/plots.py
src/
  generate_v2.py      builds the items from the anchors
  realizations.py     the surface wordings each event can take
  modal_runner.py     runs the models: probe, generation, all framings
  run_plan.py         emits the run plan; run_all.sh executes it
  parsing.py          the answer parser, mirrored inside the runner
  preflight.py        the pre-run gate
  test_parsing.py     parser tests and the drift check
  judge_traces.py     blinded trace judge
  judge_choice.py     blinded choice judge
  rescore.py          re-parse a stored run without re-running it
  analyze.py          summarise any single results CSV
  analyze_human.py    the human ceiling, gated on the annotator's own checks
  check_fillers.py    the filler-adjacency control
  make_annotation.py  build and score an annotation set
  check_dataset.py    what the related benchmarks record, and where
  check_hb_events.py  the HorizonBench overlap measurement
  check_realpref.py   the RealPref overlap measurement
  mine_horizonbench.py  the attempt to source items from HorizonBench
  figs/plots.py       the figures
  figs/trace_patterns.py  the trace justification patterns
```

`figures/` and `results/` are written by the code rather than tracked, so a
fresh clone will not have them.

## Two readouts

Models are scored two ways, because no single readout covers all nine.

The **forced-choice probe** compares next-token probabilities for the two
option letters. It is valid only where a model puts most of its mass on those
letters, which the Qwen models do and Gemma-3 does not.

**Free-text generation** is scored by a parser with a documented precedence
order, and under the natural framing by a blinded judge instead, because there
is no answer tag to parse. Eight of nine models return a readable answer on
93.3% to 100% of replies. Qwen3.5-0.8B copies the answer template and stops on
69.2% of replies, so it is reported but not interpreted under generation.

Where both readouts apply they agree, and both are reported.

## Citation

```bibtex
@misc{BIBKEY,
  title  = {TITLE},
  author = {Zarreen Reza},
  year   = {2026},
  eprint = {ARXIV_ID},
  archivePrefix = {arXiv}
}
```

## Licence

Apache 2.0. Every item is generated by our own code from preference
attributes and events written for this study, so no third-party data licence
applies. Model and tooling licences are listed in the paper's appendix.
