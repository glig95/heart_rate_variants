"""
Method 3: the machine learning model (XGBoost).

The question it answers: taken together, how well do these variants predict a
species' heart rate, and which variants does the model make most use of?

This method is different in kind from the other two, in two ways that matter.

First, it does not correct for the phylogeny. PGLS and the clade-sharing method
both discount the fact that related species resemble each other; this model does
not, so a variant can score highly simply because it marks a group of relatives
that happen to share a heart rate. The leave-one-clade-out check below is what
exposes that, but it works at the level of the whole model, not variant by
variant.

Second, it produces importance rather than a p-value. Importance says how much a
variant improved the model's fit; it is not a test against chance, so the
p-value column is left empty on purpose.

Many parameter combinations are tried and the best one is kept, so nothing
has to be chosen by hand. The search scores each combination with whole clades
held out, five blocks at a time so that it does not take all day; the winner is
then scored properly, in both of these ways:

  random split          species are split into groups at random. This scores
                        higher, because a test species usually has a close
                        relative in the training set.

  leave-one-clade-out   whole clades are held out in turn, so the model has to
                        predict a lineage it has never seen.

The difference between the two is itself a result worth reading.
"""

from itertools import product
import numpy as np

from . import stats_tools
from .method_pgls import _direction


def parameter_grid(size, seed=0):
    """
    Build the list of parameter combinations to try.

    Six things are varied: the number of trees, how deep each tree may go, how
    fast the model learns, how many variants are screened in before fitting, what
    fraction of the species each tree sees, and how strongly large weights are
    penalized. Every combination of those is 648 in all, which is more than is
    worth fitting, so `size` of them are drawn.

    The draw is shuffled with a fixed seed rather than taken in order, because
    the ordered list would cover the first parameter thoroughly and the last one
    barely. A fixed seed means the same combinations are tried every time, so a
    run can be repeated exactly.
    """
    trees = [200, 300, 500]
    depths = [2, 3, 4]
    rates = [0.01, 0.03, 0.1]
    features = [100, 300, 600, 1000]
    subsamples = [0.6, 0.8, 1.0]
    penalties = [1.0, 5.0]

    combinations = [dict(n_estimators=t, max_depth=d, learning_rate=r, n_features=f,
                         subsample=u, reg_lambda=p)
                    for t, d, r, f, u, p in product(trees, depths, rates, features,
                                                    subsamples, penalties)]
    if size >= len(combinations):
        return combinations
    order = np.random.default_rng(seed).permutation(len(combinations))
    return [combinations[i] for i in order[:size]]


def _make_model(parameters, seed):
    """Create an XGBoost regressor with one combination of parameters."""
    import xgboost

    return xgboost.XGBRegressor(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        learning_rate=parameters["learning_rate"],
        subsample=parameters.get("subsample", 0.8),
        reg_lambda=parameters.get("reg_lambda", 1.0),
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=0)


def _fit_and_predict(genotypes, target, train, test, parameters, seed):
    """
    Train on one part of the species and predict the rest.

    Everything that learns from the data, including filling uncalled genotypes,
    scoring which variants look useful, and rescaling, is worked out on the
    training species only. If any of it used the test species the accuracy would
    be too high.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression

    filler = SimpleImputer(strategy="mean").fit(genotypes[train])
    train_x = filler.transform(genotypes[train])
    test_x = filler.transform(genotypes[test])

    n_keep = min(parameters["n_features"], train_x.shape[1])
    screen = SelectKBest(f_regression, k=n_keep).fit(train_x, target[train])
    train_x, test_x = screen.transform(train_x), screen.transform(test_x)

    scaler = StandardScaler().fit(train_x)
    train_x, test_x = scaler.transform(train_x), scaler.transform(test_x)

    model = _make_model(parameters, seed).fit(train_x, target[train])
    return model.predict(test_x)


def _cross_validate(genotypes, target, groups, parameters, seed):
    """
    Predict every species while it is held out, and return those predictions.

    `groups` says which species are held out together: random groups for the
    random split, clades for leave-one-clade-out.
    """
    predictions = np.full(len(target), np.nan)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        if train.sum() >= 5 and test.sum() >= 1:
            predictions[test] = _fit_and_predict(genotypes, target, train, test,
                                                 parameters, seed)
    return predictions


def clade_blocked_folds(clade_labels, n_folds=5):
    """
    Put whole clades into a small number of folds of roughly equal size.

    Used while searching for parameters. Holding out five blocks of clades tests
    the same thing as holding out one clade at a time, since no fold ever
    contains a relative of the species it is predicting, but it fits the model
    far fewer times. The winning parameters are then scored properly, one clade
    at a time.
    """
    labels = np.asarray(clade_labels)
    sizes = sorted(((int((labels == clade).sum()), clade) for clade in np.unique(labels)),
                   reverse=True)
    load = [0] * n_folds
    fold_of = {}
    for size, clade in sizes:                 # largest clade into the emptiest fold
        smallest = int(np.argmin(load))
        fold_of[clade] = smallest
        load[smallest] += size
    return np.array([fold_of[c] for c in labels])


def search_parameters(genotypes, target, clade_labels, settings, progress=None):
    """
    Try many parameter combinations and return the best one.

    Each combination is scored by a rank correlation measured with whole clades
    held out, which is the score that says whether the model has learned anything
    beyond family resemblance. Returns the winning parameters and the full table
    of scores, which is written out with the results.
    """
    say = progress or (lambda message: None)
    grid = parameter_grid(settings.xgb_search_size)
    folds = clade_blocked_folds(clade_labels)
    scores = []

    for number, parameters in enumerate(grid, start=1):
        predictions = _cross_validate(genotypes, target, folds,
                                      parameters, settings.random_seed)
        score = stats_tools.spearman(target, predictions)
        scores.append(dict(parameters, clade_blocked_spearman=score))
        say(f"Model {number} of {len(grid)}: "
            f"{parameters['n_estimators']} trees, depth {parameters['max_depth']}, "
            f"rate {parameters['learning_rate']}, {parameters['n_features']} variants, "
            f"sample {parameters['subsample']}, penalty {parameters['reg_lambda']} "
            f"→ {score:.3f}")

    best = max(scores, key=lambda row: (row["clade_blocked_spearman"]
                                        if np.isfinite(row["clade_blocked_spearman"])
                                        else -np.inf))
    say(f"Best model: {best['n_estimators']} trees, depth {best['max_depth']}, "
        f"rate {best['learning_rate']}, {best['n_features']} variants screened in, "
        f"sample {best['subsample']}, penalty {best['reg_lambda']} "
        f"→ {best['clade_blocked_spearman']:.3f}")
    return best, scores


def run(dataset, settings, clade_labels, values=None, progress=None):
    """
    Search for the best model, score it both ways, and rank the variants.

    Returns a table with one row per variant, a summary of how accurate the best
    model was, the table of every model tried, and the inputs needed to measure
    what each variant contributes.
    """
    say = progress or (lambda message: None)
    genotypes = (dataset.genotypes.values if values is None else values).astype(float)
    target = dataset.log_heart_rate.copy()

    # When body mass is being treated as a confounder its effect has already been
    # taken out of the heart rate handed in, using the tree, so nothing is done
    # about it here.
    mass_note = (" The body-mass effect had been removed from heart rate first, so the "
                 "model predicts only what body size does not explain."
                 if settings.use_mass else "")

    say(f"Searching {settings.xgb_search_size} parameter combinations")
    best, scores = search_parameters(genotypes, target, clade_labels, settings, say)

    generator = np.random.default_rng(settings.random_seed)
    random_groups = generator.integers(0, settings.xgb_random_folds, len(target))

    say("Scoring the best model on a random split")
    random_predictions = _cross_validate(genotypes, target, random_groups,
                                         best, settings.random_seed)
    say("Scoring the best model by leaving out whole clades")
    clade_predictions = _cross_validate(genotypes, target, clade_labels,
                                        best, settings.random_seed)

    accuracy = {
        "n_models_tried": len(scores),
        "best_n_estimators": best["n_estimators"],
        "best_max_depth": best["max_depth"],
        "best_learning_rate": best["learning_rate"],
        "best_n_features": best["n_features"],
        "best_subsample": best["subsample"],
        "best_reg_lambda": best["reg_lambda"],
        "random_split_spearman": stats_tools.spearman(target, random_predictions),
        "random_split_r_squared": stats_tools.r_squared(target, random_predictions),
        "leave_one_clade_out_spearman": stats_tools.spearman(target, clade_predictions),
        "leave_one_clade_out_r_squared": stats_tools.r_squared(target, clade_predictions),
        "n_species": int(len(target)),
        "mass_removed_first": bool(settings.use_mass),
        "note": ("A random split scores higher than leaving out whole clades, because "
                 "a test species usually has a close relative in the training set. The "
                 "leave-one-clade-out score is the one that says whether sequence "
                 "predicts heart rate in a lineage the model has not seen." + mass_note),
    }

    say("Fitting the best model on all species")
    from sklearn.impute import SimpleImputer
    filled = SimpleImputer(strategy="mean").fit_transform(genotypes)
    final_model = _make_model(best, settings.random_seed).fit(filled, target)
    importance = final_model.feature_importances_.astype(float)

    # A description of which way each variant points: the difference in average
    # heart rate between the species that carry it and those that do not. This
    # is a description only; the other two methods do the testing.
    raw = dataset.genotypes.values.astype(float)
    heart_rate = dataset.log_heart_rate
    with np.errstate(invalid="ignore"):
        carriers = np.where(raw == 1, 1.0, np.nan)
        others = np.where(raw == 0, 1.0, np.nan)
        mean_carrier = np.nanmean(carriers * heart_rate[:, None], axis=0)
        mean_other = np.nanmean(others * heart_rate[:, None], axis=0)
    difference = mean_carrier - mean_other

    table = dataset.variant_info.copy()
    table["effect_percent"] = [stats_tools.percent_change_in_heart_rate(d) for d in difference]
    table["direction"] = _direction(table.effect_percent)
    table["importance"] = importance
    total = importance.sum()
    table["importance_share_percent"] = 100 * importance / total if total > 0 else np.nan
    table["n_species_tested"] = int(len(target))
    table["statistic"] = importance
    table["statistic_name"] = "XGBoost gain: how much this variant improved the model"
    table["p_value"] = np.nan
    table["significance_method"] = (
        "none. XGBoost importance ranks variants inside one model and is not a "
        "statistical test, and this method does not correct for the phylogeny. "
        "Use PGLS or the clade-sharing method for a p-value.")
    table["method"] = "XGBoost"
    table["mass_included"] = settings.use_mass
    table["effect_percent_meaning"] = (
        "plain difference in average heart rate between carriers and non-carriers, "
        "with no correction for shared ancestry; shown to give a direction only")
    return table, accuracy, scores, (genotypes, target, clade_labels, best)


def measure_contributions(table, model_inputs, settings, n_top, progress=None):
    """
    Measure how much each of the top variants adds to the model.

    For every one of them the whole leave-one-clade-out check is run again with
    that variant removed. The drop in accuracy is what the variant was worth. A
    variant with a high importance score but no drop is one the model could have
    done without, because other variants carry the same information.
    """
    say = progress or (lambda message: None)
    genotypes, target, clade_labels, best = model_inputs

    baseline = stats_tools.r_squared(
        target, _cross_validate(genotypes, target, clade_labels, best, settings.random_seed))

    ranked = table.sort_values("importance", ascending=False).head(n_top)
    drops = {}
    for count, (position, row) in enumerate(ranked.iterrows(), start=1):
        say(f"Measuring the contribution of variant {count} of {len(ranked)}: {row.variant}")
        without = np.delete(genotypes, position, axis=1)
        reduced = stats_tools.r_squared(
            target, _cross_validate(without, target, clade_labels, best, settings.random_seed))
        drops[row.variant] = baseline - reduced

    table = table.copy()
    table["variance_explained"] = table.variant.map(drops)
    table["variance_explained_meaning"] = (
        "drop in leave-one-clade-out R squared when this variant is removed from "
        "the model and it is refitted; measured for the listed variants only")
    table["model_r_squared_with_all_variants"] = baseline
    return table


def select(table, settings):
    """
    Keep the most important variants. There is no p-value to threshold on, so
    the top `xgb_n_report` variants by importance are listed.
    """
    hits = table.sort_values("importance", ascending=False).head(settings.xgb_n_report)
    return hits.reset_index(drop=True)
