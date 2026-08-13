"""
Method 1: PGLS, phylogenetic generalized least squares.

The question it answers: across the whole tree, is carrying this variant
associated with a different heart rate, once shared ancestry is accounted for?

How it works, in words:
  1. The tree gives a covariance matrix saying how much history each pair of
     species shares.
  2. That matrix is used to "whiten" heart rate and the genotypes, which means
     rescaling them so that closely related species stop counting as repeated
     measurements of the same thing.
  3. On the whitened data an ordinary regression of heart rate on the genotype
     is fitted, and its slope is tested against zero.

Given for every variant:
  effect_percent          how much heart rate differs in carriers, in percent
  variance_explained      the squared phylogeny-corrected partial correlation
  p_value                 from the t-test on the regression slope
"""

from collections import OrderedDict

import numpy as np

from . import stats_tools


def run(dataset, settings, values=None, progress=None):
    """
    Test every variant with PGLS and return one row per variant.

    `values` is the genotype matrix to use, already prepared according to the
    missing-data rule; when it is left out the raw table is used.

    When `settings.use_mass` is on the heart rate handed in has already had the
    body-mass effect taken out of it, by `data_io.remove_body_mass_effect`. This
    method does not add mass to the model itself; it only accounts for the one
    quantity that fit used up when it works out the degrees of freedom.
    """
    say = progress or (lambda message: None)
    say("Preparing the phylogeny")

    covariance = dataset.covariance.values.astype(float)
    heart_rate = dataset.log_heart_rate.copy()
    genotypes = (dataset.genotypes.values if values is None else values).astype(float)

    # Every species has a value for every variant, so one whitening serves the
    # whole table and all variants can be tested in a single pass.
    if np.isfinite(genotypes).all():
        results = _test_all_at_once(genotypes, heart_rate, covariance, settings, say)
    else:
        results = _test_one_at_a_time(genotypes, heart_rate, covariance, settings, say)

    table = dataset.variant_info.copy()
    for column, values in results.items():
        table[column] = values

    table["direction"] = _direction(table.effect_percent)
    table["method"] = "PGLS"
    table["significance_method"] = (
        "t-test on the slope of a generalized least-squares regression of "
        "log10 heart rate on the genotype, using the tree covariance matrix"
        + (", after the body-mass effect had been removed from heart rate"
           if settings.use_mass else "")
    )
    table["variance_explained_meaning"] = (
        "squared partial correlation between genotype and heart rate after "
        "removing shared ancestry"
        + (", with the body-mass effect already removed from heart rate"
           if settings.use_mass else ""))
    table["mass_included"] = settings.use_mass
    return table


def _direction(effect):
    """
    Say which way each variant points: faster, slower, or not tested at all.

    Variants that could not be tested, because too few species carry them or
    none was sequenced, get "not tested" rather than a direction that would look
    like a result.
    """
    effect = np.asarray(effect, dtype=float)
    return np.where(~np.isfinite(effect), "not tested",
                    np.where(effect >= 0, "faster", "slower"))


def _test_all_at_once(genotypes, heart_rate, covariance, settings, say):
    """
    Test every variant in one pass.

    Possible when missing genotypes have been filled in, because then every
    variant uses the same set of species and the whitening only has to be
    worked out once. This is the fast path.
    """
    say("Whitening the data")
    whitener = stats_tools.cholesky_whitener(covariance)
    n_species = len(heart_rate)
    # Usually one per species, but fewer when the tree cannot tell some species
    # apart. It is this count, not the number of species, that says how many
    # independent observations the test really has.
    n = whitener.shape[0]
    if n < n_species:
        say(f"The tree separates only {n} of the {n_species} species, so the test "
            f"is carried out on {n} independent observations")

    # A variant every species carries, or none does, holds no information and
    # would give a meaningless slope rather than an honest blank.
    varying = np.nanstd(genotypes, axis=0) > 0

    white_y = whitener @ heart_rate
    white_ones = whitener @ np.ones(n_species)
    white_x = whitener @ genotypes

    fixed = np.column_stack([white_ones])

    say("Testing every variant")
    projection = fixed @ np.linalg.pinv(fixed)
    residual_y = white_y - projection @ white_y
    residual_x = white_x - projection @ white_x

    numerator = residual_y @ residual_x
    denominator = np.sqrt((residual_y @ residual_y) *
                          np.einsum("ij,ij->j", residual_x, residual_x))
    with np.errstate(divide="ignore", invalid="ignore"):
        partial_r = np.where((denominator > 0) & varying, numerator / denominator, np.nan)

    # The mass fit estimated one quantity from the same data, so it costs a
    # degree of freedom even though mass is no longer a column here.
    extra = 1 if settings.use_mass else 0
    p_values = np.array([stats_tools.correlation_p_value(r, n, extra) for r in partial_r])

    say("Working out effect sizes")
    slopes, low, high = _slopes(white_y, white_x, fixed, varying)

    return dict(
        n_species_tested=np.full(len(partial_r), n_species),
        statistic=partial_r,
        statistic_name=["phylogeny-corrected partial correlation"] * len(partial_r),
        variance_explained=partial_r ** 2,
        p_value=p_values,
        effect_percent=[stats_tools.percent_change_in_heart_rate(s) for s in slopes],
        effect_percent_low=[stats_tools.percent_change_in_heart_rate(s) for s in low],
        effect_percent_high=[stats_tools.percent_change_in_heart_rate(s) for s in high],
    )


def _slopes(white_y, white_x, fixed, varying=None):
    """
    Fit the regression slope of each variant and its 95 percent range.

    The slope is on the log10 heart-rate scale; the caller converts it into a
    percentage. The range is the slope plus or minus the right multiple of its
    standard error for the number of species tested.
    """
    n_variants = white_x.shape[1]
    slopes = np.full(n_variants, np.nan)
    low = np.full(n_variants, np.nan)
    high = np.full(n_variants, np.nan)
    multiplier = stats_tools.t_multiplier(len(white_y) - fixed.shape[1] - 1)
    for j in range(n_variants):
        if varying is not None and not varying[j]:
            continue
        design = np.column_stack([fixed, white_x[:, j]])
        coefficients, errors, _ = stats_tools.least_squares(design, white_y)
        slopes[j] = coefficients[-1]
        low[j] = coefficients[-1] - multiplier * errors[-1]
        high[j] = coefficients[-1] + multiplier * errors[-1]
    return slopes, low, high


def _test_one_at_a_time(genotypes, heart_rate, covariance, settings, say):
    """
    Test every variant separately, dropping the species that have no call for it.

    Used when the settings say to ignore missing genotypes rather than fill
    them in. It is slower, because the whitening has to be redone for each
    different set of species.
    """
    n_variants = genotypes.shape[1]
    out = {k: np.full(n_variants, np.nan) for k in
           ["n_species_tested", "statistic", "variance_explained", "p_value",
            "effect_percent", "effect_percent_low", "effect_percent_high"]}
    # Variants often share a pattern of uncalled species, so whitenings are kept
    # and reused. Real data can hold more than a thousand different patterns and
    # each whitening is a full matrix, so the store is capped and the
    # longest-unused one is dropped when it fills.
    cache = OrderedDict()
    cache_limit = 64

    for j in range(n_variants):
        if j % 500 == 0:
            say(f"Testing variant {j} of {n_variants}")
        column = genotypes[:, j]
        present = np.isfinite(column)
        if present.sum() < 10 or np.nanstd(column[present]) == 0:
            continue
        key = present.tobytes()
        if key in cache:
            cache.move_to_end(key)
        else:
            cache[key] = stats_tools.cholesky_whitener(covariance[np.ix_(present, present)])
            if len(cache) > cache_limit:
                cache.popitem(last=False)
        whitener = cache[key]

        n_species = int(present.sum())
        n = whitener.shape[0]
        white_y = whitener @ heart_rate[present]
        white_x = whitener @ column[present]
        fixed = np.column_stack([whitener @ np.ones(n_species)])

        projection = fixed @ np.linalg.pinv(fixed)
        residual_y = white_y - projection @ white_y
        residual_x = white_x - projection @ white_x
        denominator = np.sqrt((residual_y @ residual_y) * (residual_x @ residual_x))
        if denominator == 0:
            continue
        r = float((residual_y @ residual_x) / denominator)

        coefficients, errors, _ = stats_tools.least_squares(
            np.column_stack([fixed, white_x]), white_y)
        slope, error = coefficients[-1], errors[-1]

        out["n_species_tested"][j] = n_species
        out["statistic"][j] = r
        out["variance_explained"][j] = r * r
        out["p_value"][j] = stats_tools.correlation_p_value(r, n, 1 if settings.use_mass else 0)
        multiplier = stats_tools.t_multiplier(n - fixed.shape[1] - 1)
        out["effect_percent"][j] = stats_tools.percent_change_in_heart_rate(slope)
        out["effect_percent_low"][j] = stats_tools.percent_change_in_heart_rate(
            slope - multiplier * error)
        out["effect_percent_high"][j] = stats_tools.percent_change_in_heart_rate(
            slope + multiplier * error)

    out["statistic_name"] = ["phylogeny-corrected partial correlation"] * n_variants
    return out


def select(table, settings):
    """
    Keep the variants that pass the p-value threshold, strongest first.
    """
    hits = table[table.p_value < settings.p_threshold].copy()
    return hits.sort_values("p_value").reset_index(drop=True)
