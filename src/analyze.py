#!/usr/bin/env python3
"""Re-run the analysis on a results CSV, with no GPU and no Modal.

    python analyze.py results/generate_thinkon__phase3_thinkon.csv
    python analyze.py results/probe_chat_prefill__phase1_main.csv --model Qwen/Qwen3.5-9B

WHY THIS EXISTS
---------------
`modal run` loads main() and summarize() into the local process when it starts,
so editing the analysis while a run is in flight does not change that run's
printed output. 
"""
import argparse
import os
import sys
import types


def _stub_modal():
    m = types.ModuleType("modal")

    class _I:
        @staticmethod
        def debian_slim(**k):
            return _I()

        def pip_install(self, *a, **k):
            return self

    class _V:
        @staticmethod
        def from_name(*a, **k):
            return _V()

        def commit(self):
            pass

    class _S:
        @staticmethod
        def from_dict(d):
            return d

    def _deco(f):
        return f

    class _App:
        def __init__(self, n):
            pass

        def function(self, **k):
            return _deco

        def local_entrypoint(self, **k):
            return _deco

    m.Image, m.Volume, m.Secret, m.App = _I, _V, _S, _App
    sys.modules.setdefault("modal", m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default="", help="only this model")
    ap.add_argument("--experiment", default="",
                    help="probe or generate; inferred from the columns if omitted")
    a = ap.parse_args()

    import pandas as pd
    _stub_modal()
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "mv2", os.path.join(here, "modal_runner.py"))
    mv2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mv2)

    d = pd.read_csv(a.csv)
    exp = a.experiment or ("generate" if "parse_method" in d else "probe")
    if "id" in d:
        d = d[d.id != "LOAD_FAILED"]
    models = [a.model] if a.model else sorted(d.model.unique())
    print(f"{a.csv}: {len(d)} rows, experiment={exp}, "
          f"{len(models)} model(s)")
    meta = a.csv.replace(".csv", ".meta.json")
    if os.path.exists(meta):
        import json
        print("filters:", json.dumps(json.load(open(meta)), indent=1))
    for m in models:
        g = d[d.model == m]
        if len(g):
            mv2.summarize(g, m, exp)


if __name__ == "__main__":
    main()
