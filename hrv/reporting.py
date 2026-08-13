"""
Writing the results out: tables, per-species and per-gene summaries, and the
record of which settings produced them.

Every table has the same shape whichever method made it, so results from the
three methods can be put side by side.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


# The columns every variant table carries, in the order they are written.
STANDARD_COLUMNS = [
    "rank", "variant", "gene", "position", "consensus_aa", "alt_aa", "direction",
    "effect_percent", "effect_percent_low", "effect_percent_high",
    "variance_explained", "variance_explained_meaning",
    "statistic", "statistic_name", "p_value", "significance_method",
    "n_carriers", "n_species_tested", "n_contrast_clades", "clades_with_carriers",
    "importance", "importance_share_percent",
    "method", "mass_included",
]


def make_output_folder(settings):
    """
    Create a fresh, dated folder for this run and return its path.

    The name carries the date, the method and whether body mass was included,
    so folders stay tellable apart without opening them.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    mass = "with_mass" if settings.use_mass else "no_mass"
    folder = Path(settings.output_folder) / f"{stamp}_{settings.method}_{mass}"

    # Never write into a folder that already exists, so no run can quietly
    # overwrite another.
    if folder.exists():
        number = 2
        while (folder.parent / f"{folder.name}_{number}").exists():
            number += 1
        folder = folder.parent / f"{folder.name}_{number}"

    (folder / "plots").mkdir(parents=True, exist_ok=True)
    return folder


def tidy_table(table):
    """
    Put a results table into the standard column order and drop the columns a
    given method does not fill in.
    """
    present = [c for c in STANDARD_COLUMNS if c in table.columns]
    extra = [c for c in table.columns if c not in STANDARD_COLUMNS]
    return table[present + extra]


def species_summary(dataset, hits):
    """
    Build the per-species summary: which of the kept variants each species
    carries, and how many of them point to a faster or a slower heart rate.
    """
    if hits.empty or "variant" not in hits.columns:
        hits = pd.DataFrame(columns=["variant", "direction"])
    rows = []
    faster = set(hits.loc[hits.direction == "faster", "variant"])

    for position, genome_id in enumerate(dataset.species):
        carried_fast, carried_slow, uncalled = [], [], []
        for variant in list(hits.variant):
            value = dataset.genotypes[variant].values[position]
            if not np.isfinite(value):
                uncalled.append(variant)
            elif value == 1:
                (carried_fast if variant in faster else carried_slow).append(variant)
        row = dataset.species_table.iloc[position]
        rows.append(dict(
            genome_id=genome_id,
            scientific_name=row.scientific_name,
            common_name=row.common_name,
            order=row["order"],
            heart_rate_bpm=row.heart_rate_bpm,
            body_mass_g=row.body_mass_g,
            n_faster_variants=len(carried_fast),
            n_slower_variants=len(carried_slow),
            n_not_sequenced=len(uncalled),
            faster_variants_carried="; ".join(carried_fast),
            slower_variants_carried="; ".join(carried_slow),
        ))
    return pd.DataFrame(rows).sort_values("heart_rate_bpm").reset_index(drop=True)


def gene_summary(all_variants, hits):
    """
    Build the per-gene summary: how many variants were tested in each gene, how
    many were kept, and the strongest result the gene produced.
    """
    if all_variants.empty or "gene" not in all_variants.columns:
        return pd.DataFrame(columns=["gene", "n_variants_tested", "n_variants_reported",
                                     "n_faster", "n_slower", "best_p_value",
                                     "strongest_variant"])
    rows = []
    for gene, block in all_variants.groupby("gene"):
        found = hits[hits.gene == gene]
        best = np.nanmin(block.p_value) if "p_value" in block and block.p_value.notna().any() else np.nan
        rows.append(dict(
            gene=gene,
            n_variants_tested=len(block),
            n_variants_reported=len(found),
            n_faster=int((found.direction == "faster").sum()) if len(found) else 0,
            n_slower=int((found.direction == "slower").sum()) if len(found) else 0,
            best_p_value=best,
            strongest_variant=(found.variant.iloc[0] if len(found) else ""),
        ))
    return pd.DataFrame(rows).sort_values(
        ["n_variants_reported", "best_p_value"], ascending=[False, True]).reset_index(drop=True)


def carrier_breakdown(dataset, hits, clade_labels):
    """
    Build the long table behind the plots: one row per kept variant per
    species, saying whether that species carries it and what its heart rate is.
    """
    rows = []
    names = dataset.scientific_names()
    for variant in hits.variant:
        values = dataset.genotypes[variant].values
        for position, genome_id in enumerate(dataset.species):
            value = values[position]
            rows.append(dict(
                variant=variant,
                genome_id=genome_id,
                scientific_name=names[position],
                clade=clade_labels[position],
                heart_rate_bpm=dataset.species_table.heart_rate_bpm.values[position],
                state=("not sequenced" if not np.isfinite(value)
                       else "carrier" if value == 1 else "non-carrier"),
            ))
    return pd.DataFrame(rows)


def write_settings(settings, folder, dataset_notes, extra=None):
    """
    Save the settings of this run, plus notes about the data that was loaded,
    so the run can be repeated exactly.
    """
    settings.save(Path(folder) / "settings_used.json")
    lines = ["Settings used for this run are in settings_used.json.", "",
             "How the clades were defined:", settings.describe_clades(), "",
             "Notes about the data:"]
    lines += [f"  - {note}" for note in dataset_notes]
    if extra:
        lines += ["", "Notes about the run:"] + [f"  - {note}" for note in extra]
    (Path(folder) / "run_notes.txt").write_text("\n".join(lines))
