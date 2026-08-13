"""
Exploring the data, part 2: where species sit in protein space.

The idea: two species can be compared by asking at what fraction of positions
their channel proteins differ. Doing that for every pair gives a table of
distances. Those distances live in a very high number of dimensions, so they
are flattened onto a plane in the way that distorts them least. The result is a
map: species that sit close together have similar channel proteins.

The technique is classical multidimensional scaling, which for this kind of
distance is the same idea as principal component analysis.

What to look for on the map:
  * species from the same taxonomic order clustering together means the
    proteins mostly record family history
  * an axis that lines up with heart rate means protein differences along that
    direction go with how fast the heart beats
"""

import numpy as np
import pandas as pd


def distance_matrix(genotypes):
    """
    Fraction of variable positions at which each pair of species differs.

    Only positions where both species have a call are counted, so missing data
    reduces the evidence for a pair rather than pretending they are identical.
    """
    values = np.asarray(genotypes, dtype=float)
    n = values.shape[0]
    known = np.isfinite(values)
    filled = np.where(known, values, 0.0)

    both_known = known.astype(float) @ known.astype(float).T
    same = filled @ filled.T + (known * (1 - filled)) @ (known * (1 - filled)).T
    with np.errstate(divide="ignore", invalid="ignore"):
        distances = 1 - same / both_known
    distances[~np.isfinite(distances)] = np.nanmean(distances[np.isfinite(distances)])
    np.fill_diagonal(distances, 0.0)
    return (distances + distances.T) / 2


def classical_scaling(distances, n_axes=6):
    """
    Place species on a small number of axes so their distances are preserved.

    Returns the coordinates of every species and, for each axis, the share of
    the total spread it accounts for.
    """
    n = distances.shape[0]
    squared = distances ** 2
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring @ squared @ centring
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    positive = np.clip(values, 0, None)
    share = positive / positive.sum() if positive.sum() > 0 else positive
    n_axes = min(n_axes, n - 1)
    coordinates = vectors[:, :n_axes] * np.sqrt(positive[:n_axes])
    return coordinates, share[:n_axes]


def variance_explained_by_group(coordinate, groups):
    """
    How much of the spread along one axis is explained by a grouping.

    This is eta squared from a one-way analysis of variance: a value near 1
    means species of the same group sit together on the axis and different
    groups sit apart, so the axis is mostly recording that grouping.
    """
    coordinate = np.asarray(coordinate, float)
    groups = np.asarray(groups)
    grand_mean = coordinate.mean()
    between, total = 0.0, float(((coordinate - grand_mean) ** 2).sum())
    for group in np.unique(groups):
        inside = groups == group
        between += inside.sum() * (coordinate[inside].mean() - grand_mean) ** 2
    return float(between / total) if total > 0 else np.nan


def run(dataset, settings, progress=None):
    """
    Build the protein-space map and describe each of its axes.

    Returns the coordinates of every species and a table saying, for each axis,
    how much of the overall spread it holds, how much of that is taxonomic
    order, and how strongly it tracks heart rate.
    """
    say = progress or (lambda message: None)
    say("Comparing every pair of species")
    distances = distance_matrix(dataset.genotypes.values)

    say("Flattening the comparison onto a small number of axes")
    coordinates, share = classical_scaling(distances, settings.pca_n_axes)

    orders = dataset.species_table["order"].astype(str).values
    heart_rate = dataset.log_heart_rate

    rows = []
    for axis in range(coordinates.shape[1]):
        column = coordinates[:, axis]
        correlation = np.corrcoef(column, heart_rate)[0, 1]
        rows.append(dict(
            axis=f"axis {axis + 1}",
            share_of_spread=float(share[axis]),
            explained_by_taxonomic_order=variance_explained_by_group(column, orders),
            correlation_with_heart_rate=float(correlation),
        ))
    axis_table = pd.DataFrame(rows)

    positions = pd.DataFrame(
        coordinates, columns=[f"axis_{i + 1}" for i in range(coordinates.shape[1])])
    positions.insert(0, "genome_id", dataset.species)
    positions.insert(1, "scientific_name", dataset.scientific_names())
    positions.insert(2, "order", orders)
    positions.insert(3, "heart_rate_bpm", dataset.species_table.heart_rate_bpm.values)

    return positions, axis_table
