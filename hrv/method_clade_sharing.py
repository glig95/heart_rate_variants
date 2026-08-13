"""
Method 2: the clade-sharing method.

The question it answers: does the same variant go with the same shift in heart
rate again and again, inside separate groups of related species?

Why it exists: a variant can look strongly associated with heart rate simply
because it arose once, in an ancestor of one fast-beating group, and was then
inherited by all of that group's descendants. That is one event, not evidence.
The clade-sharing method only listens to comparisons made *inside* a clade,
between close relatives that carry the variant and close relatives that do not.
A variant that keeps doing the same thing in several unrelated clades is far
harder to explain by a single accident of history.

How it works, in words:
  1. Species are put into clades (see clades.py for the rules on offer).
  2. A clade is useful for a variant only if it holds both carriers and
     non-carriers of it. Those are the "contrast clades".
  3. Inside each contrast clade the clade's own average heart rate and its own
     average genotype are subtracted, so only within-clade differences remain.
  4. The pooled correlation across all contrast clades is tested against zero.
     Subtracting one average per clade uses up information, and the test
     accounts for that: the degrees of freedom are the number of species minus
     the number of contrast clades minus one.
  5. A variant is only believed if it has at least a set number of contrast
     clades, two by default, because one clade means one evolutionary event.
"""

import numpy as np

from . import stats_tools
from .method_pgls import _direction


def run(dataset, settings, clade_labels, values=None, progress=None):
    """
    Test every variant with the clade-sharing method and return one row per variant.

    If `settings.use_mass` is on, the heart rate handed in has already had the
    body-mass effect removed from it, so the association given
    is not simply a size effect.
    """
    say = progress or (lambda message: None)
    genotypes = (dataset.genotypes.values if values is None else values).astype(float)
    heart_rate = dataset.log_heart_rate
    n_variants = genotypes.shape[1]

    columns = {name: np.full(n_variants, np.nan) for name in
               ["statistic", "variance_explained", "p_value", "effect_percent",
                "effect_percent_low", "effect_percent_high", "n_species_tested"]}
    n_contrast = np.zeros(n_variants, dtype=int)
    clade_lists = [""] * n_variants
    carrier_counts = np.zeros(n_variants, dtype=int)

    for j in range(n_variants):
        if j % 1000 == 0:
            say(f"Testing variant {j} of {n_variants}")
        result = _test_one_variant(genotypes[:, j], heart_rate,
                                   clade_labels, settings)
        n_contrast[j] = result["n_contrast_clades"]
        clade_lists[j] = result["clades"]
        carrier_counts[j] = result["n_carriers_in_contrast"]
        for name in columns:
            columns[name][j] = result[name]

    table = dataset.variant_info.copy()
    for name, column in columns.items():
        table[name] = column
    table["n_contrast_clades"] = n_contrast
    table["clades_with_carriers"] = clade_lists
    table["n_carriers_in_contrast_clades"] = carrier_counts
    table["statistic_name"] = "within-clade correlation"
    table["direction"] = _direction(table.effect_percent)
    table["method"] = "clade-sharing"
    table["significance_method"] = (
        "t-test on the correlation between genotype and log10 heart rate after "
        "subtracting each clade's own average from both, on degrees of freedom "
        "reduced by one per clade"
        + (", with the body-mass effect already removed from heart rate"
           if settings.use_mass else ""))
    table["variance_explained_meaning"] = (
        "squared within-clade correlation: the share of the heart-rate "
        "differences between close relatives that this variant tracks")
    table["mass_included"] = settings.use_mass
    return table


def _test_one_variant(genotype, heart_rate, clade_labels, settings):
    """
    Run the clade-sharing test for a single variant.

    Returns the correlation, its p-value, the effect size, how many clades
    contributed, and which ones.
    """
    empty = dict(statistic=np.nan, variance_explained=np.nan, p_value=np.nan,
                 effect_percent=np.nan, effect_percent_low=np.nan,
                 effect_percent_high=np.nan, n_species_tested=np.nan,
                 n_contrast_clades=0, clades="", n_carriers_in_contrast=0)

    present = np.isfinite(genotype)
    if present.sum() < 4:
        return empty

    x = genotype[present]
    y = heart_rate[present]
    clades = clade_labels[present]

    contrast = []
    for clade in np.unique(clades):
        inside = clades == clade
        carriers = int(np.nansum(x[inside]))
        if 0 < carriers < int(inside.sum()):
            contrast.append(clade)
    if len(contrast) == 0:
        return empty

    chosen = np.isin(clades, contrast)
    x, y, clades = x[chosen], y[chosen], clades[chosen]

    # Subtract each clade's own average, so only within-clade differences remain.
    x = x.astype(float).copy()
    y = y.astype(float).copy()
    for clade in contrast:
        inside = clades == clade
        x[inside] -= x[inside].mean()
        y[inside] -= y[inside].mean()

    n = len(y)
    if n < 3 or x.std() == 0:
        return empty

    denominator = np.sqrt((x @ x) * (y @ y))
    if denominator == 0:
        return empty
    r = float((x @ y) / denominator)
    # One average was removed per clade, and the body-mass fit that produced the
    # heart rate handed in costs one more, so both come off the degrees of freedom.
    used_up = (len(contrast) - 1) + (1 if settings.use_mass else 0)
    p_value = stats_tools.correlation_p_value(r, n, used_up)

    design = np.column_stack([np.ones(n), x])
    coefficients, errors, _ = stats_tools.least_squares(design, y)
    slope, error = coefficients[1], errors[1]

    # Write the clade list as "Rodentia(22); Primates(20)": name and carrier count.
    original = genotype[present][chosen]
    described = []
    for clade in sorted(contrast):
        inside = clades == clade
        described.append(f"{clade}({int(np.nansum(original[inside] > 0))})")

    return dict(
        statistic=r,
        variance_explained=r * r,
        p_value=p_value,
        effect_percent=stats_tools.percent_change_in_heart_rate(slope),
        effect_percent_low=stats_tools.percent_change_in_heart_rate(slope - 1.96 * error),
        effect_percent_high=stats_tools.percent_change_in_heart_rate(slope + 1.96 * error),
        n_species_tested=n,
        n_contrast_clades=len(contrast),
        clades="; ".join(described),
        n_carriers_in_contrast=int(np.nansum(original > 0)),
    )


def select(table, settings):
    """
    Keep the variants that pass the p-value threshold and appear in enough
    separate clades, strongest first.
    """
    hits = table[(table.p_value < settings.p_threshold) &
                 (table.n_contrast_clades >= settings.min_contrast_clades)].copy()
    return hits.sort_values("p_value").reset_index(drop=True)
