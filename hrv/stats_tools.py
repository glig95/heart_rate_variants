"""
Small statistical helpers shared by the methods.

Nothing here is specific to heart rate or to genetics; these are the general
pieces (whitening, correlations, p-values, effect sizes) that the three
methods build on.
"""

import numpy as np
from scipy import stats


def cholesky_whitener(covariance):
    """
    Build the matrix that removes shared ancestry from the data.

    A covariance matrix V can be written as V = L times L transposed. Anything
    multiplied by the inverse of L behaves as if the species were independent,
    which is exactly what an ordinary regression assumes. This function returns
    that inverse.

    Some trees make V impossible to split that way: two species sitting at
    exactly the same point, because the tree gives their divergence as zero,
    produce two identical rows. The tree then holds no information about how
    those two differ, and the comparison between them carries no weight at all.

    In that case the matrix is rebuilt from its eigenvalues and the flat
    directions are dropped rather than kept. Dropping them is the point. Nudging
    them up to some small number instead, which is the obvious repair, is the
    wrong one: the smaller the nudge, the more enormous the weight the method
    then places on a comparison the tree says nothing about, and a single such
    pair is enough to swamp every variant and return p-values like 1e-250.

    The returned matrix has one row per direction the tree can actually speak
    to, which is usually every species but is fewer when the tree has flat
    directions in it. Callers should take the number of rows, not the number of
    species, as the number of independent observations they have.
    """
    matrix = np.asarray(covariance, dtype=float)
    size = len(matrix)
    scale = float(np.mean(np.diag(matrix))) or 1.0
    try:
        lower = np.linalg.cholesky(matrix)
        # Cholesky can succeed on a matrix that is singular for practical
        # purposes, so the result is checked before it is trusted.
        if float(np.min(np.diag(lower))) ** 2 > scale * 1e-10:
            return np.linalg.inv(lower)
    except np.linalg.LinAlgError:
        pass

    values, vectors = np.linalg.eigh(matrix)
    keep = values > scale * 1e-10
    if not keep.any():
        raise np.linalg.LinAlgError(
            "This covariance matrix is entirely flat, so it says nothing about "
            "how the species are related.")
    return (vectors[:, keep] / np.sqrt(values[keep])).T


def correlation_p_value(r, n_observations, n_extra_terms=0):
    """
    Turn a correlation into a two-sided p-value using Student's t distribution.

    `n_extra_terms` counts any variables already taken out of the data, such as
    body mass. Degrees of freedom are the number of observations minus two,
    minus those extra terms.
    """
    degrees = n_observations - 2 - n_extra_terms
    if degrees <= 0 or not np.isfinite(r) or abs(r) >= 1:
        return np.nan
    t = r * np.sqrt(degrees / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), degrees))


def t_multiplier(degrees):
    """
    The number to multiply a standard error by for a 95 percent range.

    This is 1.96 for a large sample and more than that for a small one. Using
    1.96 everywhere would make the range too narrow when few species were
    tested.
    """
    if degrees <= 0:
        return np.nan
    return float(stats.t.ppf(0.975, degrees))


def percent_change_in_heart_rate(slope):
    """
    Convert a slope on log10 heart rate into a percentage change in beats per minute.

    Heart rate is modeled on a log10 scale, so a slope of 0.3 means heart rate
    is multiplied by 10 to the power 0.3, which is about a 100 percent increase.

    What this gives is the change in the typical, meaning the median, heart rate
    of a carrier against a non-carrier, not the change in the average of the two
    groups. On a log scale those are not the same thing.
    """
    if not np.isfinite(slope):
        return np.nan
    return float(100 * (10 ** slope - 1))


def least_squares(design, response):
    """
    Fit a straight-line model and return the coefficients, their standard errors,
    and their p-values.

    `design` has one column per term in the model, including the intercept.
    This is ordinary least squares; the phylogeny is handled by whitening the
    data before it gets here.
    """
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    # Counting the columns would overstate the degrees of freedom when two
    # columns say the same thing, which happens with a variant every species
    # carries. The rank counts only the columns that add something.
    degrees = design.shape[0] - int(np.linalg.matrix_rank(design))
    if degrees <= 0:
        nan = np.full(len(coefficients), np.nan)
        return coefficients, nan, nan
    variance = float(residuals @ residuals) / degrees
    covariance = variance * np.linalg.pinv(design.T @ design)
    errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = coefficients / errors
    p_values = 2 * stats.t.sf(np.abs(t_values), degrees)
    return coefficients, errors, p_values


def spearman(a, b):
    """Rank correlation between two lists, ignoring pairs where either is missing."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return np.nan
    return float(stats.spearmanr(a[keep], b[keep]).correlation)


def r_squared(observed, predicted):
    """
    Fraction of the variation in `observed` that `predicted` accounts for.

    Returns a number from 1 (perfect) downwards; it can go below zero when the
    predictions are worse than simply guessing the average.
    """
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    if keep.sum() < 3:
        return np.nan
    error = np.sum((observed[keep] - predicted[keep]) ** 2)
    spread = np.sum((observed[keep] - observed[keep].mean()) ** 2)
    if spread == 0:
        return np.nan
    return float(1 - error / spread)
