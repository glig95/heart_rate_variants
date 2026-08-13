"""
The combined rule: variants that PGLS and the clade-sharing method both find.

Each of the two tests has a blind spot. PGLS looks across the whole tree and
can be fooled by a variant that marks one large fast-beating lineage. The
clade-sharing method only looks inside clades and can miss a variant that is
real but rare. Asking for both to agree is stricter than either alone.

A variant is kept when:
  * PGLS gives a p-value below the threshold, and
  * the clade-sharing method gives a p-value below the threshold, and
  * the clade-sharing method saw it in at least the required number of clades.
"""

import numpy as np

from . import method_pgls, method_clade_sharing, stats_tools
from .method_pgls import _direction


def run(dataset, settings, clade_labels, pgls_values=None, clade_values=None,
        progress=None):
    """
    Run both tests and return their results side by side in one table.

    The two matrices are passed in because the two tests can want different
    treatments of uncalled genotypes; see the missing-data note in the README.
    """
    say = progress or (lambda message: None)
    say("Running PGLS")
    pgls = method_pgls.run(dataset, settings, values=pgls_values, progress=say)
    say("Running the clade-sharing method")
    sharing = method_clade_sharing.run(dataset, settings, clade_labels,
                                       values=clade_values, progress=say)

    keys = ["variant", "gene", "position", "consensus_aa", "alt_aa",
            "n_carriers", "n_species_called"]
    table = pgls[keys].copy()

    table["pgls_statistic"] = pgls.statistic.values
    table["pgls_p_value"] = pgls.p_value.values
    table["pgls_variance_explained"] = pgls.variance_explained.values
    table["pgls_effect_percent"] = pgls.effect_percent.values

    table["clade_sharing_statistic"] = sharing.statistic.values
    table["clade_sharing_p_value"] = sharing.p_value.values
    table["clade_sharing_variance_explained"] = sharing.variance_explained.values
    table["clade_sharing_effect_percent"] = sharing.effect_percent.values
    table["n_contrast_clades"] = sharing.n_contrast_clades.values
    table["clades_with_carriers"] = sharing.clades_with_carriers.values

    # The headline columns describe the variant using the whole-tree test,
    # because that is the one with a proper effect size and confidence range.
    table["statistic"] = pgls.statistic.values
    table["statistic_name"] = "phylogeny-corrected partial correlation (PGLS)"
    table["p_value"] = np.maximum(pgls.p_value.values, sharing.p_value.values)
    table["variance_explained"] = pgls.variance_explained.values
    table["variance_explained_meaning"] = (
        "squared phylogeny-corrected partial correlation from PGLS; the "
        "clade-sharing value is in its own column")
    table["effect_percent"] = pgls.effect_percent.values
    table["effect_percent_low"] = pgls.effect_percent_low.values
    table["effect_percent_high"] = pgls.effect_percent_high.values
    table["n_species_tested"] = pgls.n_species_tested.values
    table["direction"] = _direction(table.effect_percent)
    table["method"] = "combined (PGLS and clade-sharing)"
    table["significance_method"] = (
        "both tests had to pass. The p-value shown is the weaker of the two: "
        "PGLS t-test on the whole tree, and the clade-sharing t-test on "
        "within-clade differences. A variant passes only if both are below the "
        "threshold and it appears in enough separate clades.")
    table["mass_included"] = settings.use_mass
    return table


def select(table, settings):
    """
    Keep the variants that both tests agree on, strongest first.
    """
    hits = table[(table.pgls_p_value < settings.p_threshold) &
                 (table.clade_sharing_p_value < settings.p_threshold) &
                 (table.n_contrast_clades >= settings.min_contrast_clades)].copy()
    return hits.sort_values("pgls_p_value").reset_index(drop=True)
