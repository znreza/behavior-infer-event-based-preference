#!/bin/bash
set -e


# ================= PHASE 0 =================

# format validity
# gate: median P(A)+P(B) >= 0.5 for at least one format. If no format clears it for a model, that model is generation-only and must be dropped from phase 1's probe runs.
modal run modal_runner.py --experiment probecheck \
    --models "Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen2.5-7B-Instruct" \
    --conditions T2 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --max-items 40 \
    --tag phase0_fmt

# ================= PHASE 1 =================

# main: all models x all conditions
# gate: T1 high in BOTH option orders and T2 well below it. If T1 is not high the probe is broken, not the model. If a model is degenerate (one order at 100%, the other at 0) exclude it.
modal run modal_runner.py --experiment probe \
    --models "Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen2.5-7B-Instruct" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --fmt chat_prefill \
    --tag phase1_main

# main: generation, same cell
# gate: parse rate >= 90% AND the accuracy spread across parse_method under ~10 points. In v1 'tag' scored 67% and the fuzzy fallback 100% on the same run -- pooling them mixes two populations.
modal run modal_runner.py --experiment generate \
    --models "Qwen/Qwen3.5-9B,Qwen/Qwen2.5-7B-Instruct" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --no-thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 600 \
    --tag phase1_gen

# ================= PHASE 2 =================

# turn ablation
# gate: T1 stays high as turns grow. If T1 degrades too, length is hurting retrieval and T2 cannot be attributed to inference.
modal run modal_runner.py --experiment probe \
    --models "Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen2.5-7B-Instruct" \
    --conditions T0,T1,T2 \
    --turns 4,10,20,50 \
    --positions 0.6 \
    --frames requirement \
    --fmt chat_prefill \
    --tag phase2_turns

# event position ablation
# gate: if T2 declines with distance but T1 does not, the loss is retrieval-independent. If BOTH decline, the turn ablation's decline is distance, not length -- say so.
modal run modal_runner.py --experiment probe \
    --models "Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen2.5-7B-Instruct" \
    --conditions T0,T1,T2 \
    --turns 20 \
    --positions 0.3,0.6,0.9 \
    --frames requirement \
    --fmt chat_prefill \
    --tag phase2_pos

# ================= PHASE 3 =================

# reasoning on
# gate: log says 'thinking supported=True used=True' AND median n_gen_tokens is several times the reasoning-off run (v1: 838 vs 57). Do NOT use n_reason_chars > 0: split_reasoning counts everything before <answer> as reasoning, so it is >0 in 99.4% of reasoning-OFF rows too. Also check truncated < 5%: at a 900-token cap v1 truncated 20.8% of reasoning-on replies.
modal run modal_runner.py --experiment generate \
    --models "Qwen/Qwen3.5-9B" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 600 \
    --tag phase3_thinkon

# reasoning off, matched to phase 3
# gate: same items as the reasoning-on ladder row (same --conditions, --turns, --max-items and --seed, so the same subsample).
# OPTIONAL -- check the gate above before running
modal run modal_runner.py --experiment generate \
    --models "Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B" \
    --conditions T1,T2 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --no-thinking \
    --max-new-tokens 1024 \
    --sample 1 --gen-seed 0 \
    --max-items 400 \
    --tag phase3_thinkoff_ladder

# reasoning on, ladder
# gate: run ONLY if phase 3 moved T2 by more than 10 points, and pair it with the reasoning-off ladder row above.
# OPTIONAL -- check the gate above before running
modal run modal_runner.py --experiment generate \
    --models "Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B" \
    --conditions T1,T2 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 400 \
    --tag phase3_ladder_on

# ================= PHASE 5 =================

# gemma format validity
# gate: median P(A)+P(B) >= 0.5 for at least one format, per model. A model clearing none is generation-only.
modal run modal_runner.py --experiment probecheck \
    --models "google/gemma-3-1b-it,google/gemma-3-4b-it,google/gemma-3-12b-it" \
    --conditions T2 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --max-items 40 \
    --tag phase5_fmt

# gemma main: all conditions
# gate: T2 far below chance and T1 high in BOTH orders, as for Qwen. If T2 sits near 50 on Gemma the finding is family-specific and that is the paper.
modal run modal_runner.py --experiment probe \
    --models "google/gemma-3-1b-it,google/gemma-3-4b-it,google/gemma-3-12b-it" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --fmt chat_prefill \
    --tag phase5_main

# gemma deliberation, prompt-elicited
# gate: median n_gen_tokens well above the no-CoT arm. If it is not, the instruction was ignored and this duplicates the other arm.
modal run modal_runner.py --experiment generate \
    --models "google/gemma-3-12b-it" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --no-thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 600 \
    --cot 1 \
    --tag phase5_cot

# gemma deliberation, NEUTRAL cot
# gate: if neutral CoT leaves T2 near the no-CoT value, deliberation per se does not help and the directed gain is attention, not thinking -- which is the Qwen result replicated.
modal run modal_runner.py --experiment generate \
    --models "google/gemma-3-12b-it" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --no-thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 300 \
    --cot 2 \
    --tag phase5_cotneu

# gemma deliberation, control arm
# gate: same 600 items as the CoT arm.
modal run modal_runner.py --experiment generate \
    --models "google/gemma-3-12b-it" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --no-thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 600 \
    --cot 0 \
    --tag phase5_nocot

# gemma-27b format validity
# gate: expect no format to clear 0.5, as for 4b and 12b. If one does, say so -- it would mean the probe failure is scale-dependent.
modal run modal_runner.py --experiment probecheck \
    --models "google/gemma-3-27b-it" \
    --conditions T2 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --max-items 40 \
    --gpu A100-80GB \
    --batch-size 8 \
    --tag phase5xl_fmt

# gemma-27b main, requirement
# gate: T2 far below chance and T1 high in both orders, as for the rest of the ladder.
modal run modal_runner.py --experiment generate \
    --models "google/gemma-3-27b-it" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --no-thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 300 \
    --cot 0 \
    --gpu A100-80GB \
    --batch-size 8 \
    --tag phase5xl_req

# gemma-27b, incidental
# gate: paired against the requirement run item-for-item; the ids differ only by the _finc suffix.
modal run modal_runner.py --experiment generate \
    --models "google/gemma-3-27b-it" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames incidental \
    --no-thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 300 \
    --cot 0 \
    --gpu A100-80GB \
    --batch-size 8 \
    --tag phase5xl_inc

# gemma-27b, directed CoT
# gate: pair with the requirement run above, not with 12b's.
# OPTIONAL -- check the gate above before running
modal run modal_runner.py --experiment generate \
    --models "google/gemma-3-27b-it" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --no-thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 300 \
    --cot 1 \
    --gpu A100-80GB \
    --batch-size 8 \
    --tag phase5xl_cot

# gemma-27b, neutral CoT
# gate: expect null, as at 12b (-1.7, p=0.69).
# OPTIONAL -- check the gate above before running
modal run modal_runner.py --experiment generate \
    --models "google/gemma-3-27b-it" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --no-thinking \
    --max-new-tokens 2048 \
    --sample 1 --gen-seed 0 \
    --max-items 300 \
    --cot 2 \
    --gpu A100-80GB \
    --batch-size 8 \
    --tag phase5xl_cotneu

# ================= PHASE 6 =================

# incidental setup framing
# gate: compare T2 against phase 1 item-for-item: the ids differ only by the _finc suffix, so the contrast is paired.
modal run modal_runner.py --experiment probe \
    --models "Qwen/Qwen3.5-0.8B,Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen2.5-7B-Instruct" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames incidental \
    --fmt chat_prefill \
    --tag phase6_incidental

# ================= PHASE 4 =================

# paraphrase holdout
# gate: condition pattern matches phase 1 within a few points.
modal run modal_runner.py --experiment probe \
    --models "Qwen/Qwen3.5-9B" \
    --conditions T0,T1,T2,T3,T4 \
    --turns 4 \
    --positions 0.6 \
    --frames requirement \
    --fmt chat_prefill \
    --realization 2 \
    --tag phase4_real2

# style vs life family
# gate: report separately; do not pool.
# style vs life family: no run needed -- split the phase 1 CSV
