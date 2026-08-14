"""
Running the pipeline without the window.

Copy this file, change the settings you care about, and run it with:

    python examples/run_from_command_line.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hrv.config import Settings
from hrv import pipeline


def main():
    """Run one analysis with the settings below and print a short summary."""
    settings = Settings()

    # ---- change whatever you need here ----
    settings.method = "combined"          # "pgls", "clade_sharing", "combined", "xgboost"
    settings.use_mass = False             # True treats body mass as a confounder
    settings.clade_definition = "order"   # "order", "merged_orders", "tree_cut", "custom_file"
    settings.missing_data = "nearest"     # "nearest", "ignore", "average"
    settings.p_threshold = 1e-3
    # ---------------------------------------

    folder, hits, everything, summary = pipeline.find_variants(settings, progress=print)

    print(f"\n{len(hits)} variants passed out of {len(everything)} tested.")
    print(f"Results are in: {folder}")
    if len(hits):
        columns = ["rank", "variant", "gene", "direction", "effect_percent",
                   "variance_explained", "p_value"]
        print(hits[[c for c in columns if c in hits.columns]].head(20).to_string(index=False))


def describe_the_data():
    """
    The descriptive analyses, which are not in the window: how many variants were
    called, how much is missing, dN/dS per gene, and a map of species by protein
    similarity. Call this instead of main() to run them.
    """
    settings = Settings()
    folder, results = pipeline.explore_data(settings, progress=print)
    print(f"\nWritten to: {folder}")
    for item in results.get("made", []):
        print("  ", item)


if __name__ == "__main__":
    main()
    # describe_the_data()
