"""
Deciding which species belong to which clade.

The clade-sharing method asks whether a variant is associated with heart rate
*inside* groups of related species. That only means something once the groups
are fixed, so this file holds every way of making them. The window shows the
user, in words, exactly which rule is in force before anything is run.
"""

import numpy as np
import pandas as pd

from . import trees


def clades_from_orders(species_table):
    """
    Group species by taxonomic order (Rodentia, Carnivora, Chiroptera, ...).

    Orders come from the species file, not from the data being tested, which
    is what makes them a fair grouping.
    """
    return dict(zip(species_table.genome_id, species_table["order"].astype(str)))


def clades_from_merged_orders(species_table, min_size=3):
    """
    Group species by taxonomic order, then absorb the very small orders.

    Any order with fewer than `min_size` species is merged into the order
    whose median heart rate is closest to its own. Because the merging looks at
    heart rate, the trait under test, it is a less conservative rule than plain
    orders.
    """
    orders = species_table["order"].astype(str).values
    heart_rate = np.log10(species_table.heart_rate_bpm.values.astype(float))

    names, counts = np.unique(orders, return_counts=True)
    small = set(names[counts < min_size])
    large = [n for n in names if n not in small]
    median = {n: np.median(heart_rate[orders == n]) for n in names}

    merged = orders.copy()
    for name in small:
        if not large:
            continue
        nearest = min(large, key=lambda big: abs(median[big] - median[name]))
        merged[orders == name] = nearest
    return dict(zip(species_table.genome_id, merged))


def clades_from_tree_cut(tree_file, n_clades):
    """
    Group species by cutting the phylogenetic tree into a chosen number of groups.

    Each group is one whole branch of the tree. The number of groups is a free
    choice: fewer, larger groups give a more forgiving test, more, smaller
    groups a stricter one.
    """
    root = trees.read_newick(tree_file)
    return trees.cut_tree_into_clades(root, n_clades)


def clades_from_file(path):
    """
    Read the user's own grouping from a .csv file.

    The file needs two columns: genome_id and clade. Any species not listed is
    left out of the clade-sharing test.
    """
    table = pd.read_csv(path)
    columns = {c.lower().strip(): c for c in table.columns}
    id_column = columns.get("genome_id") or list(table.columns)[0]
    clade_column = columns.get("clade") or list(table.columns)[1]
    return dict(zip(table[id_column].astype(str), table[clade_column].astype(str)))


def build_clades(settings, species_table):
    """
    Return the clade of every species, following the rule chosen in the settings.

    This is the one function the rest of the code calls; it hides which of the
    four rules above is in force.
    """
    if settings.clade_definition == "order":
        return clades_from_orders(species_table)
    if settings.clade_definition == "merged_orders":
        return clades_from_merged_orders(species_table, settings.min_clade_size)
    if settings.clade_definition == "tree_cut":
        return clades_from_tree_cut(settings.tree_file, settings.n_tree_clades)
    if settings.clade_definition == "custom_file":
        return clades_from_file(settings.custom_clade_file)
    raise ValueError(f"Unknown clade definition: {settings.clade_definition}")


def clade_array(settings, species_table, species_order):
    """
    Return the clade of each species as a plain array, lined up with the
    genotype table's species order. Species with no clade are labeled "none".
    """
    lookup = build_clades(settings, species_table)
    return np.array([str(lookup.get(s, "none")) for s in species_order])


def summarize_clades(clade_labels):
    """
    Return a small table of clade name and how many species it holds,
    used to show the user what the chosen rule actually produced.
    """
    names, counts = np.unique(clade_labels, return_counts=True)
    table = pd.DataFrame(dict(clade=names, n_species=counts))
    return table.sort_values("n_species", ascending=False).reset_index(drop=True)
