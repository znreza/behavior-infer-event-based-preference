#!/usr/bin/env python3
"""
The experiment plan. Every run is declared here with its purpose, so nothing is
run ad hoc and nothing gets re-run because a factor was forgotten.

    python run_plan.py                 # show the matrix and cost
    python run_plan.py --emit          # emit the shell commands in order
    python run_plan.py --emit --phase 1

DESIGN
------
Four factors, and each run varies ONE while holding the others fixed:

  model      Qwen3.5 0.8B / 2B / 4B / 9B   scale
             Qwen2.5-7B-Instruct           different family, non-reasoning
  condition  T0 T1 T2 T3 T4                where in the chain it breaks
  turns      4 / 10 / 20 / 50              conversation length
  event_pos  0.3 / 0.6 / 0.9               distance from event to question
  mode       probe / gen-think / gen-plain  measurement method and deliberation

Both generation arms sample at T=0.6 / top_p=0.95 / top_k=20 with a fixed seed
-- Qwen's documented thinking-mode setting. Greedy decoding loops in thinking
mode, and matching the decoding across arms keeps the reasoning flag the only
difference.

turns and event_pos are separate factors on purpose: holding event_pos fixed
while turns grows moves length AND distance together, so neither can be
attributed alone. The reference cell pins event_pos at 0.6; phase 2 moves each
of them with the other held fixed.

The full crossing is 5 x 5 x 4 x 3 and is not worth running. The plan below
fixes a reference cell -- all conditions, 4 turns -- and moves one factor at a
time out of it. That is what makes each result attributable.

PHASES, in dependency order. Do not start a phase until the previous one passes.

  0  validity     is the measurement meaningful for each model at all
  1  main         all models x all conditions, at the reference turn count
  2  turns        turn ablation, on the models that showed the effect
  3  reasoning    thinking on vs off, same model, same items
  4  robustness   held-out paraphrase, and the style/life split

Phase 0 gates 1. Phase 1 gates 2 and 3. Nothing in 2-4 is worth running if the
condition effect is not present in 1.
"""

import argparse, math

# --- models ---------------------------------------------------------------
LADDER = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B",
          "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
CROSS_FAMILY = ["Qwen/Qwen2.5-7B-Instruct"]      # non-reasoning, different family
ALL_MODELS = LADDER + CROSS_FAMILY
BIGGEST = "Qwen/Qwen3.5-9B"

# A second family, as its own scale ladder. Gemma 3 has NO thinking mode and
# no enable_thinking flag, so deliberation there is elicited by prompt
# (--cot 1) rather than by a template flag. Repos are GATED: HF_TOKEN must be
# set and the licence accepted, or every load 401s. 27B needs an 80GB card.
GEMMA = ["google/gemma-3-1b-it", "google/gemma-3-4b-it",
         "google/gemma-3-12b-it"]
GEMMA_BIGGEST = "google/gemma-3-12b-it"
# 12B in bfloat16 is ~24.5GB and Gemma 3 uses sliding-window attention on 5 of
# every 6 layers, so the KV cache stays small. It fits the default 40GB card;
# 27B (~54GB) would have needed A100-80GB and is out of scope.
GEMMA_GPU = ""

# The 27B completes the Gemma ladder to four points, which is what makes the
# scaling claim hard to dismiss. It needs its own row because it needs its own
# card: 54GB of bfloat16 weights plus up to 18.5GB of KV at batch 16 does not
# fit 80GB, so batch 8. Kept separate from the GEMMA rows so re-emitting the
# plan does not re-run 1b/4b/12b, which are already done.
GEMMA_XL = ["google/gemma-3-27b-it"]
GEMMA_XL_GPU = "A100-80GB"
GEMMA_XL_BS = 8

ALL_COND = "T0,T1,T2,T3,T4"
REF_TURNS = "4"                                   # reference cell
ALL_TURNS = "4,10,20,50"
REF_POS = "0.6"                                   # reference event position
REF_FRAME = "requirement"                         # reference setup framing
ALL_POS = "0.3,0.6,0.9"
POS_TURNS = "20"                                  # length held fixed here

RUNNER = "src/modal_runner.py"               # the file that actually exists

# 77 anchor events x 3 paraphrases = 231 items per
# (condition, turns, event_pos) cell in data/items_v2.json
ITEMS_PER_CELL = 231
FWD_PER_ITEM = 2                                  # both option orders


def n_items(conds, turns, positions=REF_POS, frames=REF_FRAME):
    return (ITEMS_PER_CELL * len(conds.split(",")) * len(turns.split(","))
            * len(positions.split(",")) * len(frames.split(",")))


PLAN = [
    # ---- phase 0 --------------------------------------------------------
    dict(phase=0, name="format validity", tag="phase0_fmt",
         why="A raw completion prompt makes an instruct model emit newline "
             "first, so the option letters hold almost no probability and the "
             "forced choice measures nothing. Must be checked PER MODEL: the "
             "best format is not guaranteed to be the same across families.",
         exp="probecheck", models=ALL_MODELS,
         conds="T2", turns=REF_TURNS, max_items=40,
         gate="median P(A)+P(B) >= 0.5 for at least one format. If no format "
              "clears it for a model, that model is generation-only and must "
              "be dropped from phase 1's probe runs."),

    # ---- phase 1 --------------------------------------------------------
    dict(phase=1, name="main: all models x all conditions", tag="phase1_main",
         why="The core result. Same items, same turn count, every model. This "
             "is what the paper's main table comes from.",
         exp="probe", models=ALL_MODELS,
         conds=ALL_COND, turns=REF_TURNS, max_items=0,
         gate="T1 high in BOTH option orders and T2 well below it. If T1 is "
              "not high the probe is broken, not the model. If a model is "
              "degenerate (one order at 100%, the other at 0) exclude it."),

    dict(phase=1, name="main: generation, same cell", tag="phase1_gen",
         why="The probe measures the immediate answer. This measures the "
             "answer after the model can write. Two methods agreeing on the "
             "same items is what makes the headline robust. Shares the token "
             "cap and the decoding settings with the reasoning-on row, so the "
             "thinking flag is the only difference; v1 put the p99 "
             "reasoning-off reply at 622 tokens, so 2048 does not bind here.",
         exp="generate", thinking=False, models=[BIGGEST] + CROSS_FAMILY,
         conds=ALL_COND, turns=REF_TURNS, max_items=600, max_new_tokens=2048,
         gate="parse rate >= 90% AND the accuracy spread across parse_method "
              "under ~10 points. In v1 'tag' scored 67% and the fuzzy fallback "
              "100% on the same run -- pooling them mixes two populations."),

    # ---- phase 2 --------------------------------------------------------
    dict(phase=2, name="turn ablation", tag="phase2_turns",
         why="Rules out 'the model simply could not find the event'. Run on "
             "the whole ladder, not one model: if the decline with length is "
             "scale-dependent that is itself the finding. Qwen2.5 is included "
             "because it is in the main table, so its T2 floor needs the same "
             "retrieval control -- otherwise a reviewer can ask whether the "
             "cross-family result is a retrieval artifact.",
         exp="probe", models=ALL_MODELS,
         conds="T0,T1,T2", turns=ALL_TURNS, max_items=0,
         gate="T1 stays high as turns grow. If T1 degrades too, length is "
              "hurting retrieval and T2 cannot be attributed to inference."),

    dict(phase=2, name="event position ablation", tag="phase2_pos",
         why="Separates distance-to-question from conversation length. The "
             "turn ablation above holds event_pos at 0.6, so the event's "
             "distance from the question grows with length and the two are "
             "collinear. Here length is FIXED at 20 turns and only the "
             "position moves: 0.3/0.6/0.9 put the event at turn 6/12/18, i.e. "
             "14/8/2 turns before the question.",
         exp="probe", models=ALL_MODELS,
         conds="T0,T1,T2", turns=POS_TURNS, pos=ALL_POS, max_items=0,
         gate="if T2 declines with distance but T1 does not, the loss is "
              "retrieval-independent. If BOTH decline, the turn ablation's "
              "decline is distance, not length -- say so."),

    # ---- phase 3 --------------------------------------------------------
    dict(phase=3, name="reasoning on", tag="phase3_thinkon",
         why="Does deliberation close the gap? Same model and items as "
             "reasoning-off, one flag different -- the only clean way to "
             "isolate reasoning. Comparing Qwen2.5 to Qwen3.5 would confound "
             "reasoning with size, family and training data.",
         exp="generate", thinking=True, models=[BIGGEST],
         conds=ALL_COND, turns=REF_TURNS, max_items=600,
         max_new_tokens=2048,
         gate="log says 'thinking supported=True used=True' AND median "
              "n_gen_tokens is several times the reasoning-off run (v1: 838 vs "
              "57). Do NOT use n_reason_chars > 0: split_reasoning counts "
              "everything before <answer> as reasoning, so it is >0 in 99.4% "
              "of reasoning-OFF rows too. Also check truncated < 5%: at a "
              "900-token cap v1 truncated 20.8% of reasoning-on replies."),

    dict(phase=3, name="reasoning off, matched to phase 3", tag="phase3_thinkoff_ladder",
         why="The 2B and 4B reasoning-ON runs below have no reasoning-OFF "
             "counterpart in this plan -- phase 1's generation row only covers "
             "9B and Qwen2.5. Without this row a deliberation delta cannot be "
             "computed at 2B or 4B, only an absolute accuracy.",
         exp="generate", thinking=False,
         models=["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"],
         conds="T1,T2", turns=REF_TURNS, max_items=400, max_new_tokens=1024,
         gate="same items as the reasoning-on ladder row (same --conditions, "
              "--turns, --max-items and --seed, so the same subsample).",
         optional=True),

    dict(phase=3, name="reasoning on, ladder", tag="phase3_ladder_on",
         why="Only if phase 3 shows an effect at 9B. Tells you whether "
             "deliberation helps more with scale.",
         exp="generate", thinking=True, models=["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"],
         conds="T1,T2", turns=REF_TURNS, max_items=400, max_new_tokens=2048,
         gate="run ONLY if phase 3 moved T2 by more than 10 points, and pair it "
              "with the reasoning-off ladder row above.",
         optional=True),

    # ---- phase 5: second family ------------------------------------------
    dict(phase=5, name="gemma format validity", tag="phase5_fmt",
         why="Phase 0 again, for the new family. The best prompt format is "
             "not guaranteed to transfer: raw scored 0.98 on Qwen2.5 and 0.08 "
             "on Qwen3.5-4B, so it must be measured per family, not assumed.",
         exp="probecheck", models=GEMMA,
         conds="T2", turns=REF_TURNS, max_items=40, gpu=GEMMA_GPU,
         gate="median P(A)+P(B) >= 0.5 for at least one format, per model. A "
              "model clearing none is generation-only."),

    dict(phase=5, name="gemma main: all conditions", tag="phase5_main",
         why="Is the effect a Qwen artifact? Four of the five models so far "
             "are Qwen3.5. This is the same reference cell on an independent "
             "family and its own scale ladder.",
         exp="probe", models=GEMMA,
         conds=ALL_COND, turns=REF_TURNS, max_items=0, gpu=GEMMA_GPU,
         gate="T2 far below chance and T1 high in BOTH orders, as for Qwen. "
              "If T2 sits near 50 on Gemma the finding is family-specific and "
              "that is the paper."),

    dict(phase=5, name="gemma deliberation, prompt-elicited", tag="phase5_cot",
         why="Gemma 3 has no thinking mode -- its chat template ACCEPTS "
             "enable_thinking and ignores it, rendering an identical prompt -- "
             "so deliberation is elicited by "
             "prompt. Run WITH and WITHOUT the step-by-step instruction on "
             "the largest model, everything else fixed. Tests deliberation "
             "itself rather than one vendor's implementation of it.",
         exp="generate", thinking=False, cot=1, models=[GEMMA_BIGGEST],
         conds=ALL_COND, turns=REF_TURNS, max_items=600, max_new_tokens=2048,
         gpu=GEMMA_GPU,
         gate="median n_gen_tokens well above the no-CoT arm. If it is not, "
              "the instruction was ignored and this duplicates the other arm."),

    dict(phase=5, name="gemma deliberation, NEUTRAL cot", tag="phase5_cotneu",
         why="The directed CoT instruction names the event and asks what it "
             "implies, which hands over the task structure -- a gain under it "
             "is not evidence about deliberation. This arm asks only 'Think "
             "step by step before answering', which is what Qwen's native "
             "thinking mode supplies. Three-way: none / neutral / directed.",
         exp="generate", thinking=False, cot=2, models=[GEMMA_BIGGEST],
         conds=ALL_COND, turns=REF_TURNS, max_items=300, max_new_tokens=2048,
         gate="if neutral CoT leaves T2 near the no-CoT value, deliberation "
              "per se does not help and the directed gain is attention, not "
              "thinking -- which is the Qwen result replicated."),

    dict(phase=5, name="gemma deliberation, control arm", tag="phase5_nocot",
         why="The matched no-CoT arm for the row above. Same items, same "
             "seed, same decoding; only the step-by-step instruction differs.",
         exp="generate", thinking=False, cot=0, models=[GEMMA_BIGGEST],
         conds=ALL_COND, turns=REF_TURNS, max_items=600, max_new_tokens=2048,
         gpu=GEMMA_GPU,
         gate="same 600 items as the CoT arm."),

    # ---- phase 5b: complete the Gemma ladder -----------------------------
    dict(phase=5, name="gemma-27b format validity", tag="phase5xl_fmt",
         why="Completes the 'no prompt format makes the probe valid for "
             "Gemma' claim. 4b scored 0.223 and 12b 0.095 under chat_prefill, "
             "both wanting to emit ' **'. Asserting it for the family without "
             "testing the largest member would be an unforced gap.",
         exp="probecheck", models=GEMMA_XL,
         conds="T2", turns=REF_TURNS, max_items=40,
         gpu=GEMMA_XL_GPU, batch_size=GEMMA_XL_BS,
         gate="expect no format to clear 0.5, as for 4b and 12b. If one does, "
              "say so -- it would mean the probe failure is scale-dependent."),

    dict(phase=5, name="gemma-27b main, requirement", tag="phase5xl_req",
         why="The fourth point on the Gemma ladder. Same 300 items, same "
             "method, same decoding as 1b/4b/12b.",
         exp="generate", thinking=False, cot=0, models=GEMMA_XL,
         conds=ALL_COND, turns=REF_TURNS, max_items=300, max_new_tokens=2048,
         gpu=GEMMA_XL_GPU, batch_size=GEMMA_XL_BS,
         gate="T2 far below chance and T1 high in both orders, as for the "
              "rest of the ladder."),

    dict(phase=5, name="gemma-27b, incidental", tag="phase5xl_inc",
         why="THE point of adding 27b. The framing effect grew with scale in "
             "both families -- Qwen +4.8/+6.3/+22.9/+27.5 and Gemma "
             "+8.3/+13.3/+35.8. A fourth Gemma point either continues that "
             "trend or breaks it, and either way the scaling claim stops "
             "resting on three points.",
         exp="generate", thinking=False, cot=0, models=GEMMA_XL,
         conds=ALL_COND, turns=REF_TURNS, max_items=300, max_new_tokens=2048,
         frames="incidental", gpu=GEMMA_XL_GPU, batch_size=GEMMA_XL_BS,
         gate="paired against the requirement run item-for-item; the ids "
              "differ only by the _finc suffix."),

    dict(phase=5, name="gemma-27b, directed CoT", tag="phase5xl_cot",
         why="Extends the deliberation contrast to the largest model. "
             "Optional: the three-way result is already established at 12b "
             "(none 9.2, neutral 7.5, directed 37.5).",
         exp="generate", thinking=False, cot=1, models=GEMMA_XL,
         conds=ALL_COND, turns=REF_TURNS, max_items=300, max_new_tokens=2048,
         gpu=GEMMA_XL_GPU, batch_size=GEMMA_XL_BS, optional=True,
         gate="pair with the requirement run above, not with 12b's."),

    dict(phase=5, name="gemma-27b, neutral CoT", tag="phase5xl_cotneu",
         why="The control for the row above. Only worth running if the "
             "directed arm moves T2 at 27b.",
         exp="generate", thinking=False, cot=2, models=GEMMA_XL,
         conds=ALL_COND, turns=REF_TURNS, max_items=300, max_new_tokens=2048,
         gpu=GEMMA_XL_GPU, batch_size=GEMMA_XL_BS, optional=True,
         gate="expect null, as at 12b (-1.7, p=0.69)."),

    # ---- phase 6: is the setup's modal force doing the work? -------------
    dict(phase=6, name="incidental setup framing", tag="phase6_incidental",
         why="The setup reads as a stated requirement -- 'what works for me "
             "is X' -- and the finding is that models treat requirements as "
             "immutable, so a reviewer can object that the instruction-ness "
             "was built into the stimulus. This run reports the same state as "
             "a habit instead ('Lately I have mostly been going with X for my "
             "Y'), holding the noun, state, event and options fixed. Either "
             "T2 improves, and the claim sharpens to a phrasing effect, or it "
             "does not, and the objection is closed.",
         exp="probe", models=ALL_MODELS,
         conds=ALL_COND, turns=REF_TURNS, max_items=0, frames="incidental",
         gate="compare T2 against phase 1 item-for-item: the ids differ only "
              "by the _finc suffix, so the contrast is paired."),

    # ---- phase 4 --------------------------------------------------------
    dict(phase=4, name="paraphrase holdout", tag="phase4_real2",
         why="Each anchor has three paraphrases of its event. If the effect "
             "survives on a single held-out paraphrase, it is not carried by "
             "the wording of one realization. This replaces the author-only "
             "run: every anchor is now author-validated, so `source` is "
             "uniform and there is nothing left to split on.",
         exp="probe", models=[BIGGEST],
         conds=ALL_COND, turns=REF_TURNS, max_items=0, attributes="",
         realization=2,
         gate="condition pattern matches phase 1 within a few points."),

    dict(phase=4, name="style vs life family", tag="phase4_split",
         why="Style attributes concern how the assistant replies, life "
             "attributes what the user does. v1 hinted they differ. Same run, "
             "split at analysis time -- no extra compute, listed so the split "
             "is not forgotten.",
         exp="ANALYSIS-ONLY", models=[], conds=ALL_COND, turns=REF_TURNS,
         max_items=0,
         gate="report separately; do not pool."),
]


def cost(p):
    if p["exp"] in ("ANALYSIS-ONLY",):
        return 0, 0
    n = n_items(p["conds"], p["turns"], p.get("pos", REF_POS),
                p.get("frames", REF_FRAME))
    if p.get("realization") is not None:
        n = round(n / 3)                   # three paraphrases per anchor
    if p.get("max_items"):
        n = min(n, p["max_items"])
    if p["exp"] == "probecheck":
        n = min(n, p.get("max_items", 40))
        passes = n * 4                       # four formats
    else:
        passes = n * FWD_PER_ITEM
    total = passes * len(p["models"])
    # rough: probe ~25/s batched; generation ~0.35/s at a 900-token cap, and
    # roughly inversely proportional to the cap above that
    # Generation rate is dominated by how often a sequence runs to the cap,
    # not by the cap itself: generate() stops only when the LONGEST sequence in
    # the batch is done. This is a rough figure -- trust the per-batch ETA the
    # runner prints instead.
    rate = (25 if p["exp"] in ("probe", "probecheck")
            else 0.35 * 900 / p.get("max_new_tokens", 2048))
    return total, total / rate / 60


def emit(p):
    if p["exp"] == "ANALYSIS-ONLY":
        return [f"# {p['name']}: no run needed -- split the phase 1 CSV"]
    cmds = []
    m = ",".join(p["models"])
    c = [f"modal run {RUNNER} --experiment {p['exp']}",
         f'--models "{m}"', f"--conditions {p['conds']}",
         f"--turns {p['turns']}", f"--positions {p.get('pos', REF_POS)}",
         f"--frames {p.get('frames', REF_FRAME)}"]
    if p["exp"] == "probe":
        c.append("--fmt chat_prefill")
    if p["exp"] == "generate":
        c.append("--thinking" if p.get("thinking") else "--no-thinking")
        c.append(f"--max-new-tokens {p.get('max_new_tokens', 2048)}")
        # Sampling on BOTH arms. Greedy is unusable in thinking mode (Qwen's
        # own guidance; it loops and every batch runs to the token cap), and
        # if only the thinking arm sampled then the reasoning comparison would
        # differ in two factors instead of one.
        c.append(f"--sample 1 --gen-seed {p.get('gen_seed', 0)}")
    if p.get("max_items"):
        c.append(f"--max-items {p['max_items']}")
    if p.get("source"):
        c.append(f"--source {p['source']}")
    if p.get("realization") is not None:
        c.append(f"--realization {p['realization']}")
    if p.get("cot") is not None and p["exp"] == "generate":
        c.append(f"--cot {p['cot']}")
    if p.get("gpu"):
        c.append(f"--gpu {p['gpu']}")
    if p.get("batch_size"):
        c.append(f"--batch-size {p['batch_size']}")
    # Phases 1, 2 and 4 are all `probe --fmt chat_prefill`, so they collide on
    # results/probe_chat_prefill.csv. The tag is what keeps three runs.
    c.append(f"--tag {p['tag']}")
    cmds.append(" \\\n    ".join(c))
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--phase", type=int, default=-1)
    a = ap.parse_args()

    rows = [p for p in PLAN if a.phase < 0 or p["phase"] == a.phase]

    if not a.emit:
        print(f"{'ph':>2s} {'run':38s} {'models':>7s} {'passes':>9s} {'~min':>6s}")
        tot_p = tot_m = 0
        for p in rows:
            n, mins = cost(p)
            tot_p += n; tot_m += mins
            opt = " (optional)" if p.get("optional") else ""
            print(f"{p['phase']:2d} {p['name'][:38]:38s} {len(p['models']):7d} "
                  f"{n:9,d} {mins:6.0f}{opt}")
        print(f"{'':2s} {'TOTAL':38s} {'':7s} {tot_p:9,d} {tot_m:6.0f}")
        print("\nminutes are GPU-minutes summed over models; models in one run go")
        print("to separate containers in parallel, so wall-clock is lower.\n")
        for p in rows:
            print(f"--- phase {p['phase']}: {p['name']} ---")
            print(f"  why : {p['why']}")
            print(f"  gate: {p['gate']}\n")
        return

    print("#!/bin/bash\nset -e\n")
    last = None
    for p in rows:
        if p["phase"] != last:
            print(f"\n# ================= PHASE {p['phase']} =================")
            last = p["phase"]
        print(f"\n# {p['name']}")
        print(f"# gate: {p['gate']}")
        if p.get("optional"):
            print("# OPTIONAL -- check the gate above before running")
        for c in emit(p):
            print(c)


if __name__ == "__main__":
    main()
