"""
All the figures.

One style, set once at the top: white background, black text, one color for
variants that go with a faster heart rate and one for slower. Every figure
carries a legend that says what the axes are and what any marking means, so a
figure can be read on its own without the documentation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")          # draw to files, never to a screen
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


FASTER = "#B33A3A"     # variants that go with a faster heart rate
SLOWER = "#2C6FAF"     # variants that go with a slower heart rate
NEUTRAL = "#8A8F98"    # species without the variant, and background detail
INK = "#111111"        # all text

# A few familiar animals, marked on every variant figure so the heart-rate scale
# means something at a glance. Keyed on genome identifier.
REFERENCE_SPECIES = {
    "HLcerSimCot2": "rhinoceros",
    "hg38": "human",
    "mm39": "mouse",
    "HLsorAra3": "shrew",
}


def apply_style():
    """Set the look of every figure: white background, black text, no clutter."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#444444",
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
    })


apply_style()


def log_axis_ticks(axis, values, how_many=3, which="x"):
    """
    Put at least `how_many` readable numbers on a log axis.

    Left to itself a log axis labels only the powers of ten, so a panel covering
    a narrow range can end up with one number on it or none at all. This picks
    round values spread across the range actually plotted and writes them out in
    full, rather than as powers.
    """
    values = np.asarray([v for v in np.ravel(values) if np.isfinite(v) and v > 0])
    if len(values) == 0:
        return
    low, high = float(values.min()), float(values.max())
    if low == high:
        low, high = low * 0.9, high * 1.1

    nice = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9])
    candidates = np.unique(np.concatenate(
        [nice * 10.0 ** power for power in range(-2, 5)]))
    inside = candidates[(candidates >= low) & (candidates <= high)]
    if len(inside) < how_many:
        # Nothing round enough falls inside, so space them evenly instead.
        inside = np.geomspace(low, high, how_many)
    elif len(inside) > 6:
        inside = inside[np.linspace(0, len(inside) - 1, 6).astype(int)]

    setter = axis.set_xticks if which == "x" else axis.set_yticks
    setter(inside)
    formatter = axis.xaxis if which == "x" else axis.yaxis
    formatter.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    formatter.set_minor_formatter(plt.NullFormatter())
    if which == "x":
        axis.set_xlim(low * 0.92, high * 1.08)
    else:
        axis.set_ylim(low * 0.92, high * 1.08)


def color_for(direction):
    """Return the color that stands for a faster or a slower heart rate."""
    return FASTER if str(direction).lower().startswith("f") else SLOWER


def _save(figure, path):
    """Save a figure and close it, so long runs do not fill up memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written at a resolution that still looks sharp shown small in the window,
    # and holds up when the figure is opened at full size or put in a document.
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return str(path)


def _caption(figure, text, drop=0.06, below=None):
    """
    Write an explanatory note under a figure, in small gray type.

    The text is wrapped by hand rather than by matplotlib, which does not wrap
    reliably outside the axes, and `drop` moves it clear of the axis labels.

    When the figure carries a legend under the axes, pass it as `below` and the
    note is placed just beneath wherever that legend actually ended up. Guessing
    a fixed offset instead leaves either a gap or an overlap, because the legend
    moves with the number of entries and the height of the figure.
    """
    import textwrap

    width = int(16 * figure.get_size_inches()[0])
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    y = -drop
    if below is not None:
        figure.canvas.draw()
        box = below.get_window_extent()
        y = figure.transFigure.inverted().transform((0, box.y0))[1] - drop
    figure.text(0.01, y, wrapped, fontsize=8.5, color="#444444",
                ha="left", va="top")


def plot_variant_detail(dataset, row, clade_labels, path):
    """
    One figure for one variant.

    Left panel: every species as a dot, heart rate on a log scale, carriers on
    the lower row and everyone else on the upper row, with each group's median
    marked by a vertical bar.
    Right panel: the same comparison made separately inside each clade that
    holds both carriers and non-carriers, which is the evidence the
    clade-sharing method uses.
    """
    genotype = dataset.genotypes[row["variant"]].values
    heart_rate = dataset.species_table.heart_rate_bpm.values.astype(float)
    species = list(getattr(dataset, "species", []))
    color = color_for(row.get("direction", "faster"))

    carriers = np.isfinite(genotype) & (genotype == 1)
    others = np.isfinite(genotype) & (genotype == 0)

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5),
                                         gridspec_kw={"width_ratios": [1, 1.15]})
    figure.subplots_adjust(wspace=0.12)
    generator = np.random.default_rng(0)

    left.scatter(heart_rate[others], 1 + generator.uniform(-0.08, 0.08, others.sum()),
                 s=26, color=NEUTRAL, alpha=0.75, linewidths=0)
    left.scatter(heart_rate[carriers], generator.uniform(-0.08, 0.08, carriers.sum()),
                 s=34, color=color, alpha=0.9, linewidths=0)
    if others.sum():
        left.plot([np.median(heart_rate[others])] * 2, [0.82, 1.18],
                  color=NEUTRAL, lw=2.5)
    if carriers.sum():
        left.plot([np.median(heart_rate[carriers])] * 2, [-0.18, 0.18], color=color, lw=2.5)

    # Mark a few familiar animals, so the reader can place the scale without
    # having to know what 400 beats per minute looks like. Each one is colored by
    # whether it carries the variant, which answers the first question anyone
    # asks of a result: does the mouse have it?
    for genome_id, label in REFERENCE_SPECIES.items():
        if genome_id not in species:
            continue
        where = species.index(genome_id)
        rate = heart_rate[where]
        if not np.isfinite(rate):
            continue
        call = genotype[where]
        if not np.isfinite(call):
            tone = "#9AA0A6"
        elif call == 1:
            tone = color
        else:
            tone = NEUTRAL
        left.axvline(rate, color=tone, lw=1.0, ls=":", zorder=0, alpha=0.9)
        left.text(rate, 1.62, f"{label}\n{rate:.0f}", ha="center", va="bottom",
                  fontsize=8.5, color=tone, linespacing=1.3)

    left.set_xscale("log")
    log_axis_ticks(left, heart_rate[np.isfinite(heart_rate)], how_many=4)
    left.set_yticks([0, 1])
    left.set_yticklabels([f"with the variant\n(n = {carriers.sum()})",
                          f"without it\n(n = {others.sum()})"])
    left.set_ylim(-0.5, 2.0)
    left.set_xlabel("resting heart rate (beats per minute, log scale)")
    left.set_title("Heart rate of species with and without the variant", loc="left")

    rows = []
    for clade in np.unique(clade_labels):
        inside = (clade_labels == clade) & np.isfinite(genotype)
        with_it = inside & (genotype == 1)
        without = inside & (genotype == 0)
        if with_it.sum() and without.sum():
            rows.append((clade, np.median(heart_rate[without]), np.median(heart_rate[with_it]),
                         int(with_it.sum()), int(without.sum())))
    if rows:
        rows.sort(key=lambda r: r[2] / r[1])
        positions = np.arange(len(rows))
        for y, (clade, without_median, with_median, n_with, n_without) in zip(positions, rows):
            right.plot([without_median, with_median], [y, y], color="#CCCCCC", lw=2, zorder=1)
            right.scatter(without_median, y, s=45, color=NEUTRAL, zorder=2)
            right.scatter(with_median, y, s=55, color=color, zorder=3)
        right.set_yticks(positions)
        right.set_yticklabels([f"{r[0]}  ({r[3]} of {r[3] + r[4]} have it)" for r in rows],
                              fontsize=9)
        right.yaxis.tick_right()
        right.yaxis.set_label_position("right")
        right.set_xscale("log")
        log_axis_ticks(right, [r[1] for r in rows] + [r[2] for r in rows], how_many=3)
        right.set_xlabel("median resting heart rate within the clade (bpm, log scale)")
        right.set_title(f"The same comparison inside each clade ({len(rows)} clades)",
                        loc="left")
        right.invert_yaxis()
    else:
        right.text(0.5, 0.5, "No clade holds both species with and without this variant,\n"
                             "so no within-clade comparison is possible.",
                   ha="center", va="center", fontsize=11)
        right.axis("off")

    # One legend for the whole figure, below both panels, so nothing sits on top
    # of a species or a clade.
    figure.legend(handles=[
        Line2D([], [], marker="o", ls="", color=NEUTRAL, markersize=7,
               label="without the variant"),
        Line2D([], [], marker="o", ls="", color=color, markersize=7,
               label="with the variant"),
        Line2D([], [], color=NEUTRAL, lw=2.5, label="median of each group"),
        Line2D([], [], ls=":", color=NEUTRAL, lw=1.2,
               label="named animal, colored by whether it carries the variant")],
        frameon=False, fontsize=9, ncol=4, loc="upper center",
        bbox_to_anchor=(0.5, 0.03), handletextpad=0.5, columnspacing=2.0)

    effect = row.get("effect_percent", np.nan)
    p_value = row.get("p_value", np.nan)
    headline = f"{row['variant']}  —  gene {row['gene']}, protein position {row['position']}"
    detail = []
    if np.isfinite(effect):
        direction = "higher" if effect >= 0 else "lower"
        detail.append(f"species with the variant have a {abs(effect):.0f}% {direction} heart rate")
    if np.isfinite(p_value):
        detail.append(f"p = {p_value:.1e}")
    if np.isfinite(row.get("variance_explained", np.nan)):
        detail.append(f"explains {100 * row['variance_explained']:.0f}% of the variation")
    figure.suptitle(headline + ("\n" + "   ·   ".join(detail) if detail else ""),
                    fontsize=12.5, y=1.03)
    _caption(figure, "Left: one dot per species, spread vertically only so the dots do not "
                     "overlap. Right: each line joins the median heart rate of clade members "
                     "with and without the variant, for clades that contain both.",
             drop=0.16)
    return _save(figure, path)


def draw_variant_detail_job(job):
    """
    Draw one variant figure from a small package of plain data.

    Used when several figures are drawn at once in separate processes: only
    arrays and simple values are passed across, never the whole dataset.
    """
    genotype, heart_rate, clade_labels, row, path, species = job

    class _Frame:
        pass

    dataset = _Frame()
    dataset.genotypes = pd.DataFrame({row["variant"]: genotype})
    dataset.species_table = pd.DataFrame({"heart_rate_bpm": heart_rate})
    dataset.species = list(species)
    return plot_variant_detail(dataset, row, np.asarray(clade_labels), path)


def plot_manhattan(table, path, method_name, threshold=None, value_column="p_value",
                   highlight=None, n_labels=15, weaker_of_two=False):
    """
    One dot per variant, laid out gene by gene, so whole genes can be compared.

    For the two statistical methods the height is minus log10 of the p-value, so
    a higher dot means stronger evidence and the dashed line is the threshold.
    For XGBoost the height is the variant's importance, because that method
    produces no p-value.

    When both statistical methods are run together, `weaker_of_two` says so. The
    p-value drawn is then the weaker of the two the variant received, so a dot
    sits high only when PGLS and the clade-sharing method both put it there. The
    axis says as much, because that number is not the PGLS p-value in the table.
    """
    table = table.copy()
    if value_column == "p_value":
        table = table[np.isfinite(table.p_value) & (table.p_value > 0)]
        table["height"] = -np.log10(table.p_value)
        if weaker_of_two:
            y_label = "strength of evidence:  −log10 p\n(the weaker of the two methods)"
        else:
            y_label = "strength of evidence:  −log10 p"
    else:
        table = table[np.isfinite(table[value_column])]
        table["height"] = table[value_column]
        y_label = "importance to the model\n(share of the improvement in fit)"

    table = table.sort_values(["gene", "position"]).reset_index(drop=True)
    genes = list(dict.fromkeys(table.gene))
    figure, axis = plt.subplots(figsize=(14, 6))

    positions, ticks, labels = [], [], []
    cursor = 0
    for number, gene in enumerate(genes):
        block = table[table.gene == gene]
        x = np.arange(cursor, cursor + len(block))
        shade = "#3A3A3A" if number % 2 == 0 else "#9AA0A6"
        axis.scatter(x, block.height, s=9, color=shade, linewidths=0, alpha=0.75)
        ticks.append(cursor + len(block) / 2)
        labels.append(gene)
        positions.extend(x)
        cursor += len(block) + 30

    table["x"] = positions
    if threshold is not None and value_column == "p_value":
        axis.axhline(-np.log10(threshold), color="#444444", ls="--", lw=1.1)
        axis.text(cursor, -np.log10(threshold), f" p = {threshold:g}",
                  color="#444444", fontsize=9, va="bottom", ha="right")

    chosen = table[table.variant.isin(set(highlight))] if highlight is not None \
        else table.nlargest(min(15, len(table)), "height")

    # Every variant that passed gets a ring; only the strongest are named, to
    # keep the figure readable.
    for _, row in chosen.iterrows():
        axis.scatter(row.x, row.height, s=44, facecolor="none",
                     edgecolor=color_for(row.get("direction", "faster")), lw=1.4, zorder=3)
    for _, row in chosen.nlargest(min(n_labels, len(chosen)), "height").iterrows():
        axis.annotate(row.variant.split(":")[-1], (row.x, row.height),
                      textcoords="offset points", xytext=(4, 4), fontsize=8)

    if len(table):
        axis.set_ylim(top=float(table.height.max()) * 1.12)
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axis.set_ylabel(y_label)
    axis.set_xlabel("gene (alternating shades separate neighboring genes)")
    legend = [
        Line2D([], [], marker="o", ls="", color="#3A3A3A", markersize=5,
               label="one variant tested"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=FASTER,
               markersize=9, label="passed, higher heart rate"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=SLOWER,
               markersize=9, label="passed, lower heart rate"),
    ]
    if threshold is not None and value_column == "p_value":
        legend.append(Line2D([], [], ls="--", color="#444444", label="p-value threshold"))
    drawn_legend = axis.legend(handles=legend, frameon=False, fontsize=8.5,
                               loc="upper left", bbox_to_anchor=(0, -0.28), ncol=4,
                               handletextpad=0.4, columnspacing=1.6)

    note = ("A variant is circled when it met every condition for being kept, not "
            "simply when it sits above the line. ")
    if weaker_of_two:
        note += ("Each dot is the weaker of the variant's two p-values, so height here "
                 "means both methods agreed, not that either one alone was strong. The "
                 "table gives the PGLS p-value and the clade-sharing p-value separately. ")
    if threshold is not None:
        note += ("The clade-sharing method also requires a minimum number of clades "
                 "containing both carriers and non-carriers, so a variant above the line "
                 "seen in too few clades stays uncircled. ")
    note += f"Only the {n_labels} strongest are named, to keep the figure legible."
    _caption(figure, note, drop=0.02, below=drawn_legend)
    return _save(figure, path)


def plot_method_comparison(table, path, threshold, kept):
    """
    Compare what PGLS and the clade-sharing method said about the same variants.

    Left: the two effect sizes against each other, which shows whether the
    methods agree on direction and strength.
    Right: the two strengths of evidence against each other, which shows whether
    they agree on which variants stand out.
    """
    usable = table[np.isfinite(table.pgls_statistic) &
                   np.isfinite(table.clade_sharing_statistic)]
    chosen = usable.variant.isin(set(kept))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6))

    left.scatter(usable.pgls_statistic[~chosen], usable.clade_sharing_statistic[~chosen],
                 s=7, color=NEUTRAL, alpha=0.4, linewidths=0, label="variant tested")
    left.scatter(usable.pgls_statistic[chosen], usable.clade_sharing_statistic[chosen],
                 s=38, color=FASTER, linewidths=0, label="kept by both methods")
    left.axhline(0, color="#CCCCCC", lw=0.8)
    left.axvline(0, color="#CCCCCC", lw=0.8)
    agreement = np.corrcoef(usable.pgls_statistic, usable.clade_sharing_statistic)[0, 1]
    left.set_xlabel("PGLS: phylogeny-corrected partial correlation")
    left.set_ylabel("clade-sharing: within-clade correlation")
    left.set_title(f"Effect sizes agree with r = {agreement:.2f}", loc="left")
    left.legend(frameon=False, fontsize=8.5, loc="upper left",
                bbox_to_anchor=(0, -0.14), ncol=2, handletextpad=0.4)

    ok = (usable.pgls_p_value > 0) & (usable.clade_sharing_p_value > 0)
    x = -np.log10(usable.pgls_p_value[ok])
    y = -np.log10(usable.clade_sharing_p_value[ok])
    picked = chosen[ok]
    right.scatter(x[~picked], y[~picked], s=7, color=NEUTRAL, alpha=0.4, linewidths=0,
                  label="variant tested")
    right.scatter(x[picked], y[picked], s=38, color=FASTER, linewidths=0,
                  label="kept by both methods")
    line = -np.log10(threshold)
    right.axhline(line, color="#444444", ls="--", lw=1.0)
    right.axvline(line, color="#444444", ls="--", lw=1.0,
                  label=f"threshold p = {threshold:g}")
    strength = np.corrcoef(x, y)[0, 1]
    right.set_xlabel("PGLS:  −log10 of the p-value")
    right.set_ylabel("clade-sharing:  −log10 of the p-value")
    right.set_title(f"Strength of evidence agrees with r = {strength:.2f}", loc="left")
    right.legend(frameon=False, fontsize=8.5, loc="upper left",
                 bbox_to_anchor=(0, -0.14), ncol=3, handletextpad=0.4)

    figure.suptitle("How the two methods compare on the same variants", fontsize=12.5, y=1.02)
    _caption(figure, "Only variants both methods could test are shown. A variant in the "
                     "top-right corner of the right-hand panel passed both thresholds; "
                     "kept variants also had to appear in enough separate clades.")
    return _save(figure, path)


def plot_hits_per_gene(hits, path, method_name):
    """
    How many kept variants each gene contributed, split by whether the
    variant goes with a higher or a lower heart rate.
    """
    if hits.empty:
        return None
    if "direction" not in hits.columns:
        hits = hits.assign(direction="faster")
    counts = hits.groupby(["gene", "direction"]).size().unstack(fill_value=0)
    for column in ("faster", "slower"):
        if column not in counts:
            counts[column] = 0
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=True).index]

    height = max(3, 0.42 * len(counts) + 1.8)
    figure, axis = plt.subplots(figsize=(8, height))
    y = np.arange(len(counts))
    axis.barh(y, counts.faster, color=FASTER, label="higher heart rate in carriers")
    axis.barh(y, counts.slower, left=counts.faster, color=SLOWER,
              label="lower heart rate in carriers")
    axis.set_yticks(y)
    axis.set_yticklabels(counts.index)
    axis.set_xlabel("number of variants kept")
    axis.set_ylabel("gene")
    # Counts are whole numbers, so the axis should never offer halves.
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_title(f"Which genes the variants came from  —  {method_name}", loc="left")
    # Below the axes rather than above it, where it would sit on the title. The
    # offset is worked out in inches and then converted, so a tall figure does
    # not push the legend further away than a short one.
    legend = axis.legend(frameon=False, fontsize=8.5, loc="upper left",
                         bbox_to_anchor=(0, -0.75 / (0.77 * height)),
                         ncol=2, handletextpad=0.4)
    _caption(figure, "Genes with no variant kept are left out. Bar length is a count, "
                     "not an effect size.", drop=0.02, below=legend)
    return _save(figure, path)


def plot_species_summary(summary, path, method_name):
    """
    One bar per species: how many of the kept variants it carries, with the
    species ordered by their own heart rate.
    """
    if summary.empty:
        return None
    table = summary.sort_values("heart_rate_bpm")
    height = max(4, 0.19 * len(table) + 2.2)
    figure, axis = plt.subplots(figsize=(9, height))
    y = np.arange(len(table))
    axis.barh(y, table.n_faster_variants, color=FASTER,
              label="variants linked to a higher heart rate")
    axis.barh(y, table.n_slower_variants, left=table.n_faster_variants,
              color=SLOWER, label="variants linked to a lower heart rate")
    axis.set_yticks(y)
    axis.set_yticklabels([f"{n}  ({h:.0f} bpm)" for n, h in
                          zip(table.scientific_name, table.heart_rate_bpm)], fontsize=6.5)
    axis.set_xlabel("number of kept variants carried")
    axis.set_ylabel("species, slowest heart rate at the top")
    axis.set_title(f"What each species carries  —  {method_name}", loc="left")
    # Below the axes. Above it the legend would sit on the title.
    legend = axis.legend(frameon=False, fontsize=8.5, loc="upper left",
                         bbox_to_anchor=(0, -0.75 / (0.77 * height)),
                         ncol=2, handletextpad=0.4)
    axis.invert_yaxis()
    _caption(figure, "Each species is labeled with its own resting heart rate in brackets. "
                     "A species can appear with no bar because it carries none of the "
                     "kept variants, or because it was not sequenced at those "
                     "positions.", drop=0.02, below=legend)
    return _save(figure, path)


def plot_model_accuracy(accuracy, path):
    """
    How well the best machine learning model (XGBoost) predicted heart rate, under a
    random split and under leaving out whole clades.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    labels = ["random split", "leave one clade out"]
    values = [accuracy["random_split_spearman"], accuracy["leave_one_clade_out_spearman"]]
    axis.bar(labels, values, color=[NEUTRAL, FASTER], width=0.55)
    lowest = min(0.0, min(v for v in values if np.isfinite(v)) - 0.1)
    for x, value in enumerate(values):
        axis.text(x, value + 0.02 if value >= 0 else value - 0.06,
                  f"{value:.2f}", ha="center", fontsize=11)
    axis.set_ylim(lowest, 1)
    axis.axhline(0, color="#444444", lw=0.8)
    axis.set_ylabel("agreement between predicted and observed heart rate\n"
                    "(rank correlation; 1 is perfect, 0 is no better than chance)")
    axis.set_xlabel("how the species were split for testing")
    axis.set_title("How well the best model predicts heart rate", loc="left")
    _caption(figure, "A random split lets the model see close relatives of every test "
                     "species, so it scores higher. Leaving out whole clades does not, and "
                     "is the harder test. This method does not correct for the phylogeny.")
    return _save(figure, path)


def plot_importance(hits, path):
    """The variants the machine learning model (XGBoost) made most use of, strongest first."""
    if hits.empty:
        return None
    table = hits.sort_values("importance").tail(25)
    figure, axis = plt.subplots(figsize=(8, max(3.5, 0.34 * len(table) + 2)))
    y = np.arange(len(table))
    axis.barh(y, table.importance_share_percent, color=NEUTRAL)
    axis.set_yticks(y)
    axis.set_yticklabels(table.variant, fontsize=8)
    axis.set_xlabel("share of the model's total improvement in fit (%)")
    axis.set_ylabel("variant")
    axis.set_title("Variants the model made most use of", loc="left")
    _caption(figure, "Importance is not a statistical test and carries no p-value. It says "
                     "how much the model's fit improved because of this variant, not how "
                     "unlikely that improvement would be by chance.")
    return _save(figure, path)


def plot_dnds(gene_table, path):
    """
    Selection on each gene: how strongly the protein is being kept unchanged.

    A bar to the left of 1 means changes to the protein are being removed by
    selection, which is the usual state for a gene under constraint.
    """
    height = max(4, 0.38 * len(gene_table) + 2)
    figure, axis = plt.subplots(figsize=(8, height))
    y = np.arange(len(gene_table))
    axis.barh(y, gene_table.median_omega, color=NEUTRAL)
    axis.axvline(1.0, color=FASTER, ls="--", lw=1.2, label="dN/dS = 1, no net selection")
    axis.set_yticks(y)
    axis.set_yticklabels(gene_table.gene, fontsize=9)
    axis.set_xlabel("dN/dS  (median across species)")
    axis.set_ylabel("gene")
    axis.set_title("How strongly each gene's protein is conserved", loc="left")
    legend = axis.legend(frameon=False, fontsize=8.5, loc="upper left",
                         bbox_to_anchor=(0, -0.75 / (0.77 * height)))
    axis.invert_yaxis()
    _caption(figure, "dN/dS compares changes that alter the protein with changes that do "
                     "not. Below 1 means protein changes are being removed by selection; "
                     "above 1 means they are being favored.", drop=0.02, below=legend)
    return _save(figure, path)


def plot_variant_counts(variant_info, path):
    """
    How many variants were called, and how they are spread across the genes.

    Bars are counts of variants per gene; the title carries the total and the
    number of genes.
    """
    counts = variant_info.groupby("gene").size().sort_values()
    figure, axis = plt.subplots(figsize=(8, max(4, 0.38 * len(counts) + 2)))
    y = np.arange(len(counts))
    axis.barh(y, counts.values, color=NEUTRAL)
    for position, value in zip(y, counts.values):
        axis.text(value + max(counts.values) * 0.01, position, str(value),
                  va="center", fontsize=8)
    axis.set_yticks(y)
    axis.set_yticklabels(counts.index, fontsize=9)
    axis.set_xlabel("number of variants called")
    axis.set_ylabel("gene")
    axis.set_title(f"{int(counts.sum())} variants in {len(counts)} genes", loc="left")
    _caption(figure, "A variant is one alternative amino acid at one position, kept only if "
                     "enough species carry it and enough species were sequenced there. A "
                     "long gene has more positions and so tends to yield more variants.")
    return _save(figure, path)


def plot_pca(positions, axis_table, path):
    """
    The map of species by protein similarity, colored two ways.

    Left: colored by taxonomic order, which shows whether the proteins mostly
    record family history. Right: colored by heart rate, which shows whether
    any direction on the map goes with how fast the heart beats.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 6))
    x, y = positions.axis_1.values, positions.axis_2.values

    orders = positions["order"].astype(str).values
    common = pd.Series(orders).value_counts()
    big = list(common[common >= 3].index)
    palette = plt.get_cmap("tab20")
    for number, order in enumerate(big):
        inside = orders == order
        left.scatter(x[inside], y[inside], s=34, color=palette(number % 20),
                     label=order, linewidths=0, alpha=0.9)
    rest = ~np.isin(orders, big)
    if rest.any():
        left.scatter(x[rest], y[rest], s=26, color="#CCCCCC",
                     label="orders with fewer than 3 species", linewidths=0)
    left.legend(frameon=False, fontsize=7, ncol=3, loc="upper left",
                bbox_to_anchor=(0, -0.12), title="taxonomic order")
    left.set_title("Colored by taxonomic order", loc="left")

    heart_rate = positions.heart_rate_bpm.values.astype(float)
    dots = right.scatter(x, y, c=np.log10(heart_rate), cmap="coolwarm", s=36, linewidths=0)
    bar = figure.colorbar(dots, ax=right)
    bar.set_label("resting heart rate (bpm)")
    ticks = [10, 30, 100, 300, 800]
    bar.set_ticks(np.log10(ticks))
    bar.set_ticklabels([str(t) for t in ticks])
    right.set_title("Colored by heart rate", loc="left")

    share1 = 100 * axis_table.share_of_spread.iloc[0]
    share2 = 100 * axis_table.share_of_spread.iloc[1] if len(axis_table) > 1 else float("nan")
    for panel in (left, right):
        panel.set_xlabel(f"axis 1  ({share1:.0f}% of the spread)")
        panel.set_ylabel(f"axis 2  ({share2:.0f}% of the spread)")

    figure.suptitle("Species placed by how similar their channel proteins are",
                    fontsize=12.5, y=1.03)
    _caption(figure, "Distance on the map is the fraction of variable positions at which two "
                     "species differ, flattened onto two axes with as little distortion as "
                     "possible. Species close together have similar proteins. The axes have "
                     "no units and their direction carries no meaning on its own.")
    return _save(figure, path)


def plot_relabeled_variants(relabeling, path):
    """
    What the missing-data rule did with the missing variants.

    A variant is missing in a species when that species has no amino acid called
    at the position. Missing variants are either given a value or left out of
    the tests, depending on the rule chosen on the Data settings tab, and this
    says how many of each there were and where they fell.

    Left: one bar per species, the forty most affected, since a species missing
    from several genes is the usual reason a table has holes in it. Right: one
    bar per gene. Both are percentages of that species' or that gene's own
    variants, so a small gene and a large one can be compared directly.
    """
    rule = relabeling["rule"]
    filled = relabeling["n_relabeled"] > 0
    column = "percent_relabeled" if filled else "percent_missing"
    count_column = "n_relabeled" if filled else "n_missing"
    if rule == "nearest":
        what = "given the closest species' value"
    elif rule == "average":
        what = "given the variant's average"
    else:
        what = "left out of the tests"

    per_species, per_gene = relabeling["per_species"], relabeling["per_gene"]
    worst = per_species.nlargest(min(40, len(per_species)), column)
    worst = worst.sort_values(column)
    genes = per_gene.sort_values(column)

    height = max(5.5, 0.22 * len(worst) + 2.4)
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, height),
                                         gridspec_kw=dict(width_ratios=[1, 1]))

    y = np.arange(len(worst))
    left.barh(y, worst[column], color=NEUTRAL)
    left.set_yticks(y)
    left.set_yticklabels(worst.scientific_name, fontsize=7)
    left.set_xlabel(f"percent of this species' variants {what}")
    heading = ("Species with the most missing variants"
               if len(worst) < len(per_species) else "Missing variants per species")
    left.set_title(f"{heading} ({len(worst)} of {len(per_species)})", loc="left")

    y = np.arange(len(genes))
    right.barh(y, genes[column], color=NEUTRAL)
    for position, (percent, count) in enumerate(zip(genes[column], genes[count_column])):
        right.text(percent + 0.3, position, f"{percent:.1f}%  ({count:,})",
                   va="center", fontsize=7)
    right.set_yticks(y)
    right.set_yticklabels(genes.gene, fontsize=8)
    right.set_xlabel(f"percent of this gene's variants {what}")
    right.set_title("Missing variants per gene", loc="left")
    if len(genes):
        right.set_xlim(0, max(1.0, float(genes[column].max()) * 1.35))

    total = relabeling["n_variants_across_species"]
    share = 100 * relabeling["n_missing"] / max(1, total)
    note = (f"A variant is missing in a species when that species has no amino acid "
            f"called at the position, usually because the gene is absent from its "
            f"assembly or the region did not align. Counting every variant separately "
            f"in every species gives {total:,} of them, and {relabeling['n_missing']:,} "
            f"were missing, which is {share:.1f} percent. ")
    if filled:
        note += (f"Under the rule chosen, all {relabeling['n_relabeled']:,} of the missing "
                 f"ones were {what}, and {relabeling['n_relabeled_as_carrier']:,} were "
                 f"written in as present. A species or a gene high in these bars rests "
                 f"more on values that were worked out than on values that were "
                 f"sequenced.")
    else:
        note += ("Under the rule chosen none of them were given a value; each test simply "
                 "drops the species it has no call for, so a bar here is the share of that "
                 "species or gene that took no part in the tests.")
    _caption(figure, note, drop=0.015)
    return _save(figure, path)


def plot_missing_calls(per_species, per_gene, path):
    """
    How much of the genotype table is uncalled, by species and by gene.

    Left: one bar per species, the worst forty, since a species missing from
    many genes is the usual reason a table has holes in it. Right: one bar per
    gene. Both are percentages of that species' or gene's own cells, so a small
    gene and a large one can be compared directly.
    """
    worst = per_species.nlargest(min(40, len(per_species)), "percent_missing")
    worst = worst.sort_values("percent_missing")
    genes = per_gene.sort_values("percent_missing")

    height = max(5.5, 0.22 * len(worst) + 2)
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, height),
        gridspec_kw=dict(width_ratios=[1, 1]))

    y = np.arange(len(worst))
    left.barh(y, worst.percent_missing, color=NEUTRAL)
    left.set_yticks(y)
    left.set_yticklabels(worst.scientific_name, fontsize=7)
    left.set_xlabel("percent of this species' calls missing")
    title = ("Species with the most missing calls"
             if len(worst) < len(per_species) else "Missing calls per species")
    left.set_title(f"{title} ({len(worst)} of {len(per_species)})", loc="left")

    y = np.arange(len(genes))
    right.barh(y, genes.percent_missing, color=NEUTRAL)
    for position, value in zip(y, genes.percent_missing):
        right.text(value + 0.3, position, f"{value:.1f}", va="center", fontsize=7)
    right.set_yticks(y)
    right.set_yticklabels(genes.gene, fontsize=8)
    right.set_xlabel("percent of this gene's calls missing")
    right.set_title("Missing calls per gene", loc="left")

    _caption(figure, "A variant is missing in a species when that species has no amino "
                     "acid called at that position, usually because the gene is absent "
                     "from its assembly or the region did not align. What happens to "
                     "the missing ones is set by the missing-data rule on the Data "
                     "settings tab.")
    return _save(figure, path)


# How many separate clades the carriers of a combination span. One clade means
# the combination could be a single event in one lineage; several means it has
# turned up more than once.
ONE_CLADE = "#9A5B00"
FEW_CLADES = "#8A8F98"
MANY_CLADES = "#1B6B3A"


def _clade_color(n_clades):
    """Amber for one clade, gray for a few, green for many."""
    if n_clades <= 1:
        return ONE_CLADE
    return FEW_CLADES if n_clades <= 3 else MANY_CLADES


def cooccurrence_table(dataset, hits, clade_labels, max_variants=25):
    """
    Work out which sets of variants the species actually carry.

    One row per set: how many species carry exactly that set and nothing else
    from the list, how many variants are in it, how many clades those species
    span, and which variants they are. A set of one variant counts the species
    carrying it in isolation.

    Variants stay in the order the results table gives them, strongest first, so
    the figure and the table read the same way. With more variants than
    `max_variants` only the strongest are considered, since the figure stops
    being readable long before that.
    """
    from collections import Counter

    names = list(hits.variant)[:max_variants]
    if len(names) < 2:
        return None, names
    carried = (dataset.genotypes[names].values == 1)
    clades = np.asarray(clade_labels)

    counts, clades_of = Counter(), {}
    for row, clade in zip(carried, clades):
        key = tuple(np.where(row)[0])
        if key:
            counts[key] += 1
            clades_of.setdefault(key, []).append(clade)

    rows = []
    for key, n in counts.most_common():
        rows.append(dict(
            n_species=n,
            n_variants=len(key),
            n_clades=len(set(clades_of[key])),
            clades=", ".join(sorted(set(clades_of[key]))),
            variants=", ".join(names[i] for i in key),
            _members=key,
        ))
    return pd.DataFrame(rows), names


def plot_cooccurrence(dataset, hits, clade_labels, path, max_variants=25,
                      max_columns=28):
    """
    Which variants the same species carry, and how often.

    Every column is one set of variants: the bar counts the species carrying
    exactly that set and nothing else from the list, and the dots below say
    which variants are in it. A column with a single dot is that variant on its
    own. The bars on the left are each variant's total carriers, so a variant
    whose left bar is long but whose single-dot column is short is one that
    almost always travels in company.

    Colour is how many clades the carrying species span, which is what
    separates a set that arose once in one lineage from one that has arisen
    more than once.
    """
    table, names = cooccurrence_table(dataset, hits, clade_labels, max_variants)
    if table is None or table.empty:
        return None

    n_v = len(names)
    carried = (dataset.genotypes[names].values == 1)
    totals = carried.sum(axis=0)

    # Sets that only one species carries are its own private combination and
    # would leave a long tail of columns of height one, so they are counted in
    # the note instead of drawn.
    shown = table[table.n_species >= 2]
    if shown.empty:
        shown = table
    dropped_columns = len(shown) - min(len(shown), max_columns)
    shown = shown.head(max_columns)
    private = int(table.loc[table.n_species < 2, "n_species"].sum())

    figure = plt.figure(figsize=(max(9.5, 0.42 * len(shown) + 7.5),
                                 max(6.0, 0.30 * n_v + 3.6)))
    grid = figure.add_gridspec(2, 2, width_ratios=[0.8, 3.2],
                               height_ratios=[1.15, 2.0], hspace=0.05, wspace=0.30)

    x = np.arange(len(shown))
    colors = [_clade_color(c) for c in shown.n_clades]

    top = figure.add_subplot(grid[0, 1])
    top.bar(x, shown.n_species, color=colors, width=.62)
    for xi, h in zip(x, shown.n_species):
        top.text(xi, h + shown.n_species.max() * .02, str(int(h)), ha="center", fontsize=8.5)
    top.set_ylim(0, shown.n_species.max() * 1.14)
    top.set_xlim(-0.7, len(shown) - 0.3)
    top.set_xticks([])
    top.set_ylabel("species carrying\nexactly this set")
    top.set_title("Co-occurrence pattern of identified variants", loc="left",
                  fontweight="bold")
    top.spines["bottom"].set_visible(False)

    matrix = figure.add_subplot(grid[1, 1], sharex=top)
    for row in range(0, n_v, 2):
        matrix.axhspan(row - .5, row + .5, color="#F2F3F4", zorder=0)
    for xi, (_, row) in enumerate(shown.iterrows()):
        members = list(row._members)
        color = _clade_color(row.n_clades)
        matrix.scatter([xi] * n_v, range(n_v), s=32, color="#D8DADD", zorder=1)
        matrix.plot([xi, xi], [min(members), max(members)], color=color, lw=1.6, zorder=2)
        matrix.scatter([xi] * len(members), members, s=44, color=color, zorder=3)
    matrix.set_ylim(n_v - .5, -.5)
    matrix.set_yticks(range(n_v))
    matrix.set_yticklabels([f"{i}. {v}" for i, v in enumerate(names, start=1)], fontsize=8.5)
    matrix.tick_params(axis="y", length=0, pad=6)
    matrix.set_xticks([])
    matrix.set_xlabel("each column is one set of variants;  a single dot is that variant alone",
                      fontsize=10)
    for side in ("top", "right", "bottom", "left"):
        matrix.spines[side].set_visible(False)

    left = figure.add_subplot(grid[1, 0])
    left.barh(range(n_v), totals, color=NEUTRAL, height=.55)
    for row, total in enumerate(totals):
        left.text(total + max(totals) * .02, row, str(int(total)), va="center", fontsize=8)
    left.set_ylim(n_v - .5, -.5)
    left.set_xlim(max(totals) * 1.20, 0)      # room for the number beside each bar
    left.set_yticks([])
    left.set_xlabel("species carrying\nthe variant")
    for side in ("top", "right", "left"):
        left.spines[side].set_visible(False)

    figure.legend(
        handles=[Line2D([], [], marker="s", ls="", color=ONE_CLADE, markersize=9,
                        label="1 clade"),
                 Line2D([], [], marker="s", ls="", color=FEW_CLADES, markersize=9,
                        label="2 to 3 clades"),
                 Line2D([], [], marker="s", ls="", color=MANY_CLADES, markersize=9,
                        label="4 or more clades")],
        title="clades the carrying species span", frameon=False, fontsize=9.5,
        title_fontsize=9.5, ncol=3, loc="upper right", bbox_to_anchor=(.99, 1.0))

    note = ("Variants are numbered as in the results table. A bar counts the species that "
            "carry exactly that set and nothing else from the list, so the single-dot "
            "columns are the species carrying that variant on its own. ")
    if private:
        note += f"A further {private} species each carry a set nobody else has. "
    if dropped_columns:
        note += f"{dropped_columns} further sets shared by two or more species are not drawn. "
    if len(hits) > len(names):
        note += (f"Only the {len(names)} strongest of {len(hits)} variants are shown, to keep "
                 f"the figure readable. ")
    note += "The full table is in cooccurrence.csv."
    _caption(figure, note, drop=0.10)
    return _save(figure, path)
