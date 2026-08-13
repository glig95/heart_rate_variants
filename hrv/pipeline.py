"""
The pipeline: what happens when a Run button is pressed.

Two entry points:
  find_variants(settings)  runs one method and writes every table and plot
  explore_data(settings)   runs the exploratory analyses

Both take a `progress` function. The window passes one in so the log fills up
while the work is going on; from a script it can be left out. Both return a
summary dictionary that the window turns into a message telling the user what
was produced and where it is.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import numpy as np
import pandas as pd

from . import (clades, data_io, explore_dnds, explore_pca, method_clade_sharing,
               method_combined, method_pgls, method_xgboost, plots, reporting)


METHOD_NAMES = {
    "pgls": "PGLS",
    "clade_sharing": "clade-sharing method",
    "combined": "PGLS and the clade-sharing method together",
    "xgboost": "XGBoost",
}


def _prepare_genotypes(dataset, settings, say):
    """
    Work out which genotype matrix each method should use.

    PGLS fills the missing variants, either with the variant's average or from the
    nearest species. The clade-sharing method leaves them out instead, unless
    the nearest-species rule is chosen, in which case there is nothing left to
    leave out. The machine learning model (XGBoost) fills within each training fold, so it
    takes the raw table unless the nearest-species rule filled it already.
    """
    if settings.missing_data == "nearest":
        say("Filling uncalled genotypes from the nearest species")
        filled = data_io.genotypes_for_analysis(dataset, settings, say)
        return filled, filled, filled
    if settings.missing_data == "average":
        filled = data_io.genotypes_for_analysis(dataset, settings, say)
        raw = dataset.genotypes.values.astype(float)
        return filled, raw, raw
    raw = dataset.genotypes.values.astype(float)
    return raw, raw, raw


def find_variants(settings, progress=None, dataset=None):
    """
    Run the chosen method from start to finish and write everything out.

    Returns the output folder, the variants that passed, every variant tested,
    and a summary dictionary describing what was produced.
    """
    say = progress or (lambda message: None)
    if dataset is None:
        dataset = data_io.load_dataset(settings, say)

    extra_notes = []
    if settings.use_mass:
        if int(np.sum(np.isfinite(dataset.log_mass))) < 10:
            raise ValueError(
                "Body mass was asked for as a confounder, but fewer than ten species "
                "have one. Check the body_mass_g column of the species file, or turn "
                "the body-mass option off.")
        before = len(dataset.species)
        dataset = data_io.restrict_to_species_with_mass(dataset)
        if len(dataset.species) < before:
            extra_notes.append(f"{before - len(dataset.species)} species were left out of "
                               f"this run because they have no body mass.")
        # One adjustment, shared by every method: the body-mass effect comes out
        # of heart rate here, and the methods all see the same adjusted numbers.
        dataset = data_io.remove_body_mass_effect(dataset, say)

    clade_labels = clades.clade_array(settings, dataset.species_table, dataset.species)
    clade_counts = clades.summarize_clades(clade_labels)
    say(f"Clades: {len(clade_counts)} groups, "
        f"largest {clade_counts.n_species.max()} species, "
        f"smallest {clade_counts.n_species.min()}")

    pgls_values, clade_values, model_values = _prepare_genotypes(dataset, settings, say)
    relabeling = _relabeling_summary(dataset, pgls_values, settings)
    if relabeling["n_relabeled"]:
        say(f"{relabeling['n_relabeled']} missing variants were given a value, "
            f"{relabeling['n_relabeled_as_carrier']} of them written in as present")
    else:
        say(f"{relabeling['n_missing']} missing variants were left out of the tests")

    notes = list(extra_notes)
    accuracy, model_scores = None, None

    if settings.method == "pgls":
        all_variants = method_pgls.run(dataset, settings, pgls_values, say)
        hits = method_pgls.select(all_variants, settings)

    elif settings.method == "clade_sharing":
        all_variants = method_clade_sharing.run(dataset, settings, clade_labels,
                                                clade_values, say)
        hits = method_clade_sharing.select(all_variants, settings)

    elif settings.method == "combined":
        all_variants = method_combined.run(dataset, settings, clade_labels,
                                           pgls_values, clade_values, say)
        hits = method_combined.select(all_variants, settings)

    elif settings.method == "xgboost":
        all_variants, accuracy, model_scores, model_inputs = method_xgboost.run(
            dataset, settings, clade_labels, model_values, say)
        say("Measuring what each listed variant contributes")
        all_variants = method_xgboost.measure_contributions(
            all_variants, model_inputs, settings, settings.xgb_n_report, say)
        hits = method_xgboost.select(all_variants, settings)
        notes.append("XGBoost ranks variants by importance inside one model. Importance "
                     "is not a statistical test and this method does not correct for the "
                     "phylogeny, so no p-value is given.")
        notes.append(f"{accuracy['n_models_tried']} parameter combinations were tried. "
                     f"The best scored {accuracy['leave_one_clade_out_spearman']:.2f} "
                     f"leaving out whole clades and "
                     f"{accuracy['random_split_spearman']:.2f} on a random split.")
    else:
        raise ValueError(f"Unknown method: {settings.method}")

    say(f"{len(hits)} variants passed out of {len(all_variants)} tested")

    # The output folder is made only now, so a run that fails or is stopped
    # leaves no empty folder behind.
    folder = reporting.make_output_folder(settings)

    # ------------------------------------------------------------ the tables
    hits = hits.reset_index(drop=True)
    hits.insert(0, "rank", np.arange(1, len(hits) + 1))
    reporting.tidy_table(all_variants).to_csv(folder / "all_variants_tested.csv", index=False)
    reporting.tidy_table(hits).to_csv(folder / "variants_found.csv", index=False)

    species_table = reporting.species_summary(dataset, hits)
    species_table.to_csv(folder / "summary_per_species.csv", index=False)

    genes = reporting.gene_summary(all_variants, hits)
    genes.to_csv(folder / "summary_per_gene.csv", index=False)

    if len(hits):
        reporting.carrier_breakdown(dataset, hits, clade_labels).to_csv(
            folder / "carriers_per_variant.csv", index=False)
    clade_counts.to_csv(folder / "clades_used.csv", index=False)
    relabeling["per_gene"].to_csv(folder / "missing_variants_per_gene.csv", index=False)
    relabeling["per_species"].to_csv(folder / "missing_variants_per_species.csv", index=False)
    if model_scores is not None:
        pd.DataFrame(model_scores).to_csv(folder / "all_models_tried.csv", index=False)

    # ------------------------------------------------------------- the plots
    name = METHOD_NAMES[settings.method]
    say("Drawing the overview plots")
    if settings.method == "xgboost":
        plots.plot_manhattan(all_variants, folder / "plots" / "all_variants_by_gene.png",
                             name, value_column="importance", highlight=hits.variant)
        plots.plot_model_accuracy(accuracy, folder / "plots" / "model_accuracy.png")
        plots.plot_importance(hits, folder / "plots" / "most_important_variants.png")
    else:
        plots.plot_manhattan(all_variants, folder / "plots" / "all_variants_by_gene.png",
                             name, threshold=settings.p_threshold, highlight=hits.variant,
                             weaker_of_two=(settings.method == "combined"))

    if settings.method == "combined":
        plots.plot_method_comparison(all_variants,
                                     folder / "plots" / "methods_compared.png",
                                     settings.p_threshold, set(hits.variant))

    if len(hits) >= 2:
        table, _ = plots.cooccurrence_table(dataset, hits, clade_labels)
        if table is not None:
            table.drop(columns=["_members"]).to_csv(folder / "cooccurrence.csv", index=False)
        plots.plot_cooccurrence(dataset, hits, clade_labels,
                                folder / "plots" / "variants_co_occurrence.png")

    plots.plot_relabeled_variants(relabeling, folder / "plots" / "missing_variants.png")
    plots.plot_hits_per_gene(hits, folder / "plots" / "variants_per_gene.png", name)
    plots.plot_species_summary(species_table, folder / "plots" / "variants_per_species.png",
                               name)

    _draw_variant_plots(dataset, hits, clade_labels, folder / "plots", say)

    reporting.write_settings(settings, folder, dataset.notes, notes)
    if accuracy is not None:
        pd.DataFrame([accuracy]).to_csv(folder / "model_accuracy.csv", index=False)

    say(f"Finished. Results are in {folder}")

    summary = dict(
        folder=str(folder),
        method=name,
        n_found=len(hits),
        n_tested=len(all_variants),
        n_species=len(dataset.species),
        n_clades=len(clade_counts),
        n_plots=len(hits) + 3,
        top=list(hits.variant[:5]),
        accuracy=accuracy,
    )
    return folder, hits, all_variants, summary


def _draw_variant_plots(dataset, hits, clade_labels, folder, say):
    """
    Draw one figure per variant found, several at a time.

    Each figure needs only that variant's genotypes, the heart rates and the
    clade labels, so those small pieces are handed to worker processes rather
    than the whole dataset. Falls back to drawing them one after another if
    several processes cannot be started.
    """
    if hits.empty:
        return
    if len(hits) > 200:
        say(f"Note: {len(hits)} variants passed, so {len(hits)} detail figures will be "
            f"drawn. Lower the p-value threshold for fewer.")

    heart_rate = dataset.species_table.heart_rate_bpm.values.astype(float)
    jobs = []
    for number, (_, row) in enumerate(hits.iterrows(), start=1):
        safe_name = row.variant.replace(":", "_")
        jobs.append((
            dataset.genotypes[row.variant].values,
            heart_rate,
            list(clade_labels),
            {k: row[k] for k in row.index if k in
             ("variant", "gene", "position", "direction", "effect_percent",
              "p_value", "variance_explained")},
            str(Path(folder) / f"variant_{number:03d}_{safe_name}.png"),
            list(dataset.species),
        ))

    workers = max(1, min(os.cpu_count() or 2, 8))
    say(f"Drawing {len(jobs)} variant figures using {workers} processes")
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, _ in enumerate(pool.map(plots.draw_variant_detail_job, jobs), start=1):
                if done % 10 == 0 or done == len(jobs):
                    say(f"  {done} of {len(jobs)} figures drawn")
    except Exception:
        say("Drawing the figures one at a time")
        for done, job in enumerate(jobs, start=1):
            plots.draw_variant_detail_job(job)
            if done % 10 == 0 or done == len(jobs):
                say(f"  {done} of {len(jobs)} figures drawn")


def _relabeling_summary(dataset, prepared, settings):
    """
    Count the missing variants and what the missing-data rule did with them.

    A variant is missing in a species when that species has no amino acid called
    at the position, usually because the gene is absent from its assembly or the
    region did not align. A missing variant is either given a value, which is
    what the two filling rules do, or left out of the tests, which is what the
    third rule does. Either way it is counted here, so the reader can see how
    much of the answer rests on variants that were not actually sequenced.

    Percentages are of that species' or that gene's own variants, so a gene
    present in every species and a gene present in half of them can be compared.
    """
    raw = dataset.genotypes.values.astype(float)
    missing = ~np.isfinite(raw)
    prepared = np.asarray(prepared, dtype=float)
    relabeled = missing & np.isfinite(prepared)
    genes = dataset.variant_info.gene.values
    n_variants = raw.shape[1]

    # Under the two filling rules, a missing variant that was written in as
    # present is the one that can change a result, so it is counted on its own.
    became_carrier = relabeled & (prepared > 0.5)

    per_species = pd.DataFrame(dict(
        genome_id=dataset.species,
        scientific_name=dataset.scientific_names(),
        n_variants=n_variants,
        n_missing=missing.sum(axis=1).astype(int),
        n_relabeled=relabeled.sum(axis=1).astype(int),
        n_relabeled_as_carrier=became_carrier.sum(axis=1).astype(int),
    ))
    per_species["percent_missing"] = 100 * per_species.n_missing / max(1, n_variants)
    per_species["percent_relabeled"] = 100 * per_species.n_relabeled / max(1, n_variants)
    per_species = per_species.sort_values(
        ["n_relabeled", "n_missing"], ascending=False).reset_index(drop=True)

    rows = []
    for gene in sorted(set(genes)):
        column = genes == gene
        block, filled = missing[:, column], relabeled[:, column]
        rows.append(dict(
            gene=gene,
            n_variants=int(column.sum()),
            n_variants_across_species=int(block.size),
            n_missing=int(block.sum()),
            n_relabeled=int(filled.sum()),
            n_relabeled_as_carrier=int(became_carrier[:, column].sum()),
            percent_missing=100 * block.sum() / max(1, block.size),
            percent_relabeled=100 * filled.sum() / max(1, block.size),
        ))
    per_gene = pd.DataFrame(rows).sort_values(
        ["n_relabeled", "n_missing"], ascending=False).reset_index(drop=True)

    return dict(
        rule=settings.missing_data,
        n_variants_across_species=int(raw.size),
        n_missing=int(missing.sum()),
        n_relabeled=int(relabeled.sum()),
        n_relabeled_as_carrier=int(became_carrier.sum()),
        per_species=per_species,
        per_gene=per_gene,
    )


def _missing_call_summary(dataset):
    """
    Count the genotype cells with no amino acid called, in total and broken down.

    A cell is one species at one variant. Percentages are of that species' or
    that gene's own cells, so a gene present in every species and a gene present
    in half of them can be compared directly.
    """
    values = dataset.genotypes.values.astype(float)
    uncalled = ~np.isfinite(values)
    genes = dataset.variant_info.gene.values

    per_species = pd.DataFrame(dict(
        genome_id=dataset.species,
        scientific_name=dataset.scientific_names(),
        n_variants=values.shape[1],
        n_missing=uncalled.sum(axis=1).astype(int),
        n_genes_absent=[int(sum(1 for gene in np.unique(genes)
                                if uncalled[row, genes == gene].all()))
                        for row in range(values.shape[0])],
    ))
    per_species["percent_missing"] = 100 * per_species.n_missing / max(1, values.shape[1])
    per_species = per_species.sort_values("percent_missing",
                                          ascending=False).reset_index(drop=True)

    rows = []
    for gene in sorted(set(genes)):
        block = uncalled[:, genes == gene]
        rows.append(dict(gene=gene,
                         n_variants=int(block.shape[1]),
                         n_cells=int(block.size),
                         n_missing=int(block.sum()),
                         percent_missing=100 * block.sum() / max(1, block.size),
                         n_species_absent=int((block.all(axis=1)).sum())))
    per_gene = pd.DataFrame(rows).sort_values("percent_missing",
                                              ascending=False).reset_index(drop=True)

    return dict(
        n_cells=int(uncalled.size),
        n_missing=int(uncalled.sum()),
        percent_missing=100 * float(uncalled.sum()) / max(1, uncalled.size),
        n_species_complete=int((~uncalled.any(axis=1)).sum()),
        per_species=per_species,
        per_gene=per_gene,
    )


def explore_data(settings, progress=None, dataset=None, do_counts=True,
                 do_dnds=True, do_pca=True):
    """
    Run the exploratory analyses and write their tables and plots.

    Three things can be asked for: how many variants were called and how they
    are spread across genes; dN/dS per gene, which says how strongly selection
    has preserved each protein; and a map of species by protein similarity.
    """
    say = progress or (lambda message: None)
    folder = Path(settings.output_folder) / "exploration"
    (folder / "plots").mkdir(parents=True, exist_ok=True)

    results = {}
    made = []

    if do_counts or do_pca:
        if dataset is None:
            dataset = data_io.load_dataset(settings, say)

    if do_counts:
        say("Counting the variants in each gene")
        counts = dataset.variant_info.groupby("gene").size().reset_index(name="n_variants")
        counts["share_percent"] = 100 * counts.n_variants / counts.n_variants.sum()
        counts = counts.sort_values("n_variants", ascending=False).reset_index(drop=True)
        counts.to_csv(folder / "variants_per_gene.csv", index=False)
        plots.plot_variant_counts(dataset.variant_info,
                                  folder / "plots" / "variants_per_gene.png")
        results["variant_counts"] = counts
        results["n_variants"] = int(counts.n_variants.sum())
        results["n_species"] = len(dataset.species)
        made.append("variants_per_gene.csv and its figure")

        say("Counting the calls that are missing")
        missing = _missing_call_summary(dataset)
        missing["per_species"].to_csv(folder / "missing_calls_per_species.csv", index=False)
        missing["per_gene"].to_csv(folder / "missing_calls_per_gene.csv", index=False)
        plots.plot_missing_calls(missing["per_species"], missing["per_gene"],
                                 folder / "plots" / "missing_calls.png")
        results["missing"] = missing
        made.append("missing_calls_per_species.csv, missing_calls_per_gene.csv "
                    "and their figure")

    if do_dnds:
        say("Measuring dN/dS for every gene")
        gene_table, species_table = explore_dnds.run(settings, say)
        gene_table.to_csv(folder / "dnds_per_gene.csv", index=False)
        species_table.to_csv(folder / "dnds_per_species_and_gene.csv", index=False)
        plots.plot_dnds(gene_table, folder / "plots" / "dnds_per_gene.png")
        results["dnds"] = gene_table
        made.append("dnds_per_gene.csv and its figure")

    if do_pca:
        say("Building the map of species by protein similarity")
        positions, axis_table = explore_pca.run(dataset, settings, say)
        positions.to_csv(folder / "protein_map_positions.csv", index=False)
        axis_table.to_csv(folder / "protein_map_axes.csv", index=False)
        plots.plot_pca(positions, axis_table, folder / "plots" / "protein_map.png")
        results["protein_map"] = axis_table
        made.append("protein_map_positions.csv, protein_map_axes.csv and the map figure")

    settings.save(folder / "settings_used.json")
    say(f"Finished. Exploration results are in {folder}")
    results["folder"] = str(folder)
    results["made"] = made
    return folder, results
