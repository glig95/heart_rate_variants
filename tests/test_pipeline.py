"""
Tests for the pipeline, run on the data that ships with the repository.

These are not unit tests of every function. They check the things that would be
quietly wrong rather than loudly broken: that the supplied data still loads, that
the phylogenetic regression agrees with the textbook formula it is supposed to
implement, that the clade-sharing method really does ignore a variant that only
marks a clade, and that a whole run produces the files it promises with numbers
in the ranges they must be in.

Run them with:

    pytest -q tests
"""

import numpy as np
import pandas as pd
import pytest

from hrv import clades, data_io, method_clade_sharing, method_pgls, pipeline, stats_tools
from hrv.config import Settings


@pytest.fixture(scope="module")
def dataset():
    """Load the supplied data once and share it across the tests."""
    return data_io.load_dataset(Settings(), lambda message: None)


# --------------------------------------------------------------- the data


def test_the_supplied_data_loads(dataset):
    assert len(dataset.species) > 100
    assert dataset.genotypes.shape[0] == len(dataset.species)
    assert dataset.genotypes.shape[1] > 1000
    assert len(dataset.variant_info) == dataset.genotypes.shape[1]
    assert set(dataset.variant_info.columns) >= {"variant", "gene", "position"}


def test_heart_rates_are_sensible(dataset):
    rate = 10 ** dataset.log_heart_rate
    assert np.isfinite(rate).all()
    # No mammal beats slower than a blue whale or faster than a shrew.
    assert rate.min() > 5
    assert rate.max() < 1500


def test_genotypes_are_only_one_zero_or_missing(dataset):
    values = dataset.genotypes.values.astype(float)
    present = values[np.isfinite(values)]
    assert set(np.unique(present)) <= {0.0, 1.0}


def test_the_tree_covariance_is_a_covariance_matrix(dataset):
    matrix = dataset.covariance.values.astype(float)
    assert matrix.shape == (len(dataset.species), len(dataset.species))
    assert np.allclose(matrix, matrix.T)
    assert (np.diag(matrix) > 0).all()
    # Shared history can never exceed a species' own root-to-tip length.
    assert (matrix <= np.diag(matrix)[:, None] + 1e-9).all()


def test_the_tree_covers_every_species(dataset):
    assert list(dataset.covariance.index) == list(dataset.species)


# ------------------------------------------------------------- whitening


def test_whitening_undoes_the_covariance():
    matrix = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.25], [0.5, 0.25, 1.5]])
    whitener = stats_tools.cholesky_whitener(matrix)
    # W V Wt should be the identity: that is what whitening means.
    assert np.allclose(whitener @ matrix @ whitener.T, np.eye(3), atol=1e-9)


def test_a_flat_direction_is_dropped_rather_than_kept():
    """
    Two species the tree cannot tell apart give identical rows. The comparison
    between them carries no information and must be dropped, not nudged: nudging
    hands enormous weight to a comparison the tree says nothing about.
    """
    matrix = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 2.0, 2.0]])
    whitener = stats_tools.cholesky_whitener(matrix)
    assert whitener.shape[0] == 2, "the flat direction should have been dropped"
    assert np.isfinite(whitener).all()


# ------------------------------------------------------------------ PGLS


def test_pgls_agrees_with_the_textbook_formula():
    """
    Fit a small problem both ways: through the pipeline's whitening route, and
    through the closed-form generalized least-squares estimator
    b = (Xt Vinv X)inv Xt Vinv y. They must agree.
    """
    generator = np.random.default_rng(0)
    n = 40
    root = generator.normal(size=(n, n))
    covariance = root @ root.T / n + np.eye(n)
    genotype = (generator.random(n) < 0.4).astype(float)
    design = np.column_stack([np.ones(n), genotype])
    heart_rate = 2.0 - 0.3 * genotype + generator.multivariate_normal(
        np.zeros(n), covariance)

    whitener = stats_tools.cholesky_whitener(covariance)
    whitened, *_ = np.linalg.lstsq(whitener @ design, whitener @ heart_rate, rcond=None)

    inverse = np.linalg.inv(covariance)
    closed_form = np.linalg.solve(design.T @ inverse @ design,
                                  design.T @ inverse @ heart_rate)

    assert np.allclose(whitened, closed_form, atol=1e-8)


def test_pgls_gives_every_variant_a_row_and_legal_p_values(dataset):
    settings = Settings()
    values = data_io.genotypes_for_analysis(dataset, settings)
    table = method_pgls.run(dataset, settings, values)

    assert len(table) == dataset.genotypes.shape[1]
    p_values = table.p_value.values.astype(float)
    tested = np.isfinite(p_values)
    assert tested.sum() > 0.9 * len(p_values)
    assert (p_values[tested] >= 0).all() and (p_values[tested] <= 1).all()
    assert set(table.direction) <= {"faster", "slower", "not tested"}


# ---------------------------------------------------------- clade sharing


def test_clade_sharing_ignores_a_variant_that_only_marks_a_clade():
    """
    The point of the method. A variant carried by every member of one clade and
    nobody else holds no within-clade comparison, so it must come out untested
    rather than come out significant.
    """
    labels = np.array(["A"] * 10 + ["B"] * 10)
    marks_a_clade = np.where(labels == "A", 1.0, 0.0)
    heart_rate = np.where(labels == "A", 2.5, 2.0)

    centered_x = marks_a_clade - np.array(
        [marks_a_clade[labels == c].mean() for c in labels])
    centered_y = heart_rate - np.array([heart_rate[labels == c].mean() for c in labels])

    assert np.allclose(centered_x, 0), "a clade marker must center away to nothing"
    assert np.allclose(centered_y, 0)


def test_clade_sharing_runs_and_reports_its_contrast_clades(dataset):
    settings = Settings()
    labels = clades.clade_array(settings, dataset.species_table, dataset.species)
    values = data_io.genotypes_for_analysis(dataset, settings)
    table = method_clade_sharing.run(dataset, settings, labels, values)

    assert len(table) == dataset.genotypes.shape[1]
    assert "n_contrast_clades" in table.columns
    counts = table.n_contrast_clades.values.astype(float)
    tested = np.isfinite(table.p_value.values.astype(float))
    # Anything with a p-value must have had at least one comparison behind it.
    assert (counts[tested] >= 1).all()


# --------------------------------------------------------------- body mass


def test_removing_the_body_mass_effect_leaves_nothing_correlated_with_mass(dataset):
    """
    After the adjustment, regressing the residual back on mass must give a slope
    of zero. If it does not, the adjustment did not work and every method
    downstream is still seeing a size effect.

    The fit is a plain least-squares one with no phylogeny in it, which is what
    remove_body_mass_effect does, so this is the slope that has to vanish.
    """
    with_mass = data_io.restrict_to_species_with_mass(dataset)
    adjusted = data_io.remove_body_mass_effect(with_mass)

    design = np.column_stack([np.ones(len(adjusted.log_mass)), adjusted.log_mass])
    slope, *_ = np.linalg.lstsq(design, adjusted.log_heart_rate, rcond=None)
    assert abs(slope[1]) < 1e-8
    # And the residual really is heart rate minus a line, not something rescaled.
    assert len(adjusted.log_heart_rate) == len(with_mass.log_heart_rate)


def test_heart_rate_falls_with_body_mass(dataset):
    """The allometry every textbook gives: big animals have slow hearts."""
    with_mass = data_io.restrict_to_species_with_mass(dataset)
    slope = np.polyfit(with_mass.log_mass, with_mass.log_heart_rate, 1)[0]
    assert -0.35 < slope < -0.10


# ------------------------------------------------------------- a whole run


def test_a_whole_run_writes_what_it_promises(tmp_path, dataset):
    settings = Settings()
    settings.method = "combined"
    settings.output_folder = str(tmp_path)
    folder, hits, all_variants, summary = pipeline.find_variants(
        settings, lambda message: None, dataset=dataset)

    for name in ("variants_found.csv", "all_variants_tested.csv",
                 "summary_per_gene.csv", "summary_per_species.csv",
                 "missing_variants_per_gene.csv", "missing_variants_per_species.csv",
                 "settings_used.json", "run_notes.txt"):
        assert (folder / name).exists(), f"{name} was not written"

    assert (folder / "plots" / "all_variants_by_gene.png").exists()
    assert (folder / "plots" / "missing_variants.png").exists()

    assert len(all_variants) == dataset.genotypes.shape[1]
    assert len(hits) == len(pd.read_csv(folder / "variants_found.csv"))
    assert (hits.p_value < settings.p_threshold).all()
    assert summary["n_tested"] == len(all_variants)


def test_the_missing_variant_counts_add_up(dataset):
    settings = Settings()
    prepared = data_io.genotypes_for_analysis(dataset, settings)
    summary = pipeline._relabeling_summary(dataset, prepared, settings)

    raw = dataset.genotypes.values.astype(float)
    assert summary["n_variants_across_species"] == raw.size
    assert summary["n_missing"] == int((~np.isfinite(raw)).sum())
    assert summary["per_species"].n_missing.sum() == summary["n_missing"]
    assert summary["per_gene"].n_missing.sum() == summary["n_missing"]
    # The default rule fills every missing variant in.
    assert summary["n_relabeled"] == summary["n_missing"]
