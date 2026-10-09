"""Build single-file Colab bundles: dist/measure_<model>.py.

Inlines decision_calib/models.py + decision_calib/datasets.py into the
run_measure.py template and bakes in the model name, so the VM needs only
one file (no package upload). The package remains the source of truth.
"""
import os, re, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "decision_calib")
DIST = os.path.join(HERE, "..", "dist")
os.makedirs(DIST, exist_ok=True)
DEFAULT_MODELS = ["d1-omni", "d1-3b", "opendecider-small", "jeff-2b", "jeff-0.8b"]


def read(p):
    with open(p) as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(
        description="Inline decision_calib/models.py + datasets.py into the "
                    "run_measure.py template, baking in each model name, "
                    "producing dist/measure_<model>.py (single file for Colab).")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="registered model names to build bundles for "
                         f"(default: {' '.join(DEFAULT_MODELS)})")
    args = ap.parse_args()
    template = read(os.path.join(HERE, "run_measure.py"))
    models_src = read(os.path.join(PKG, "models.py"))
    datasets_src = read(os.path.join(PKG, "datasets.py"))

    # strip the sys.path/import block; modules are inlined instead
    template = template.replace(
        'sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))\n'
        'from decision_calib import models, datasets\n',
        "# ---- inlined decision_calib/models.py ----\n" + models_src
        + "\n# ---- inlined decision_calib/datasets.py ----\n" + datasets_src
        + "\n# ---- driver (from scripts/run_measure.py) ----\n",
    )
    # module attribute access -> plain globals (no name clashes; checked)
    template = re.sub(r"\bmodels\.", "", template)
    template = re.sub(r"\bdatasets\.", "", template)

    for model in args.models:
        out = template.replace("@MODEL@", model)
        # fix the double blank line style only; keep semantics identical
        path = os.path.join(DIST, f"measure_{model}.py")
        with open(path, "w") as f:
            f.write(out)
        # sanity: it must at least compile
        import py_compile
        py_compile.compile(path, doraise=True)
        print("built", path)


if __name__ == "__main__":
    main()
