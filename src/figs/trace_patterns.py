#!/usr/bin/env python3
"""The five patterns quoted in Figure 4, with prevalence and accuracy.

    python src/figs/trace_patterns.py        # run from src/

The percentages in the Figure 4 caption come from here. The patterns are
keyword matches over the reasoning traces, so they are lower bounds on each
pattern and they are NOT mutually exclusive: a trace that states a ranking
rule and also calls the item a test matches two of them. The pairwise overlap
is printed so the caption can say how large it is. Accuracy per pattern is
printed with a Wilson interval, because the point of the figure is that these
patterns do not separate right answers from wrong ones.
"""
import pandas as pd, math, itertools
d=pd.read_csv("results/generate_thinkon__phase3_t2_full_rescored.csv",low_memory=False)
d=d[d.parsed].copy(); d["low"]=d.raw.astype(str).str.lower()
CATS={
 "1 outranks":["outweigh","takes precedence","take precedence","overrides the implied",
   "explicit preference stated","stick with","stick to the explicit","explicitly stated preference",
   "prioritise the explicit","prioritize the explicit","explicit statement wins"],
 "2 benchmark":["these prompts","these datasets","benchmark","few-shot","evaluation dataset",
   "evaluation context","rlhf","trick question","test of memory","test if the model remembers",
   "test of reading comprehension","test of retrieving","a test to see","testing whether i",
   "these types of reasoning tasks"],
 "3 derive_then_setaside":["but contradicts","would contradict","but respecting","respects the",
   "ignoring her preference","ignoring his preference","ignoring their preference"],
 "4 not_a_request":["didn't ask for a suggestion","did not ask for a suggestion",
   "he stated a preference","she stated a preference","she stated her preference",
   "he stated his preference","she stated what works","he stated what works",
   "wasn't asking for a suggestion"],
 "5 reframe_context":["usage context","not necessarily the","context, not the","describes the context",
   "is about the context"],
}
def wilson(k,n,z=1.96):
    if n==0: return (float("nan"),)*3
    p=k/n; dd=1+z*z/n; c=(p+z*z/(2*n))/dd
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/dd
    return 100*p,100*max(0,c-h),100*min(1,c+h)
N=len(d); print(f"denominator: {N} parsed traces, overall accuracy {100*d.is_correct.mean():.1f}%\n")
masks={}
for k,ws in CATS.items():
    m=d.low.apply(lambda s: any(w in s for w in ws)); masks[k]=m
    g=d[m]; a=wilson(int(g.is_correct.sum()),len(g))
    print(f"{k:24s} n={len(g):3d}  {100*len(g)/N:5.1f}% of traces   acc={a[0]:5.1f} [{a[1]:4.1f},{a[2]:5.1f}]  ({int(g.is_correct.sum())} correct)")
anym=masks["1 outranks"]
for k in list(CATS)[1:]: anym=anym|masks[k]
none=d[~anym]; a=wilson(int(none.is_correct.sum()),len(none))
print(f"\n{'matching none':24s} n={len(none):3d}  {100*len(none)/N:5.1f}%           acc={a[0]:5.1f} [{a[1]:4.1f},{a[2]:5.1f}]")
print(f"{'matching any':24s} n={int(anym.sum()):3d}  {100*anym.mean():5.1f}%")
print("\npairwise overlap (traces in both):")
for a_,b_ in itertools.combinations(CATS,2):
    print(f"  {a_:24s} & {b_:24s} {int((masks[a_]&masks[b_]).sum()):3d}")
