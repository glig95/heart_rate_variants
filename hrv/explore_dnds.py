"""
Exploring the data, part 1: dN/dS per gene.

What dN/dS measures: a DNA change inside a gene either changes the amino acid
(non-synonymous, dN) or leaves it the same (synonymous, dS). Synonymous changes
are mostly invisible to natural selection, so they accumulate at the background
rate. The ratio of the two rates, written omega, says what selection has been
doing to the gene:

  omega well below 1   changes to the protein are being removed; the gene is
                       under constraint and its sequence matters
  omega near 1         changes to the protein are neither favored nor removed
  omega above 1        changes to the protein are being favored, a sign of
                       adaptive evolution

The counting method used here is Nei and Gojobori (1986): count how many sites
in a sequence are synonymous and how many are non-synonymous, count the actual
differences of each type against the consensus sequence, and take the ratio of
the two rates after correcting for changes that happened more than once at the
same site.
"""

from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

from . import sequences


_BASES = "ACGT"


def _synonymous_site_count(codon):
    """
    How many of a codon's three positions are synonymous.

    Each position is tried with all three other bases. The fraction of those
    changes that leave the amino acid unchanged is that position's synonymous
    share; the three shares are added up. A codon therefore has between 0 and 3
    synonymous sites, and the rest of its 3 sites are non-synonymous.
    """
    amino = sequences.translate_codon(codon)
    if amino in ("X", "*"):
        return None
    synonymous = 0.0
    for position in range(3):
        same = 0
        for base in _BASES:
            if base == codon[position]:
                continue
            changed = codon[:position] + base + codon[position + 1:]
            if sequences.translate_codon(changed) == amino:
                same += 1
        synonymous += same / 3
    return synonymous


_SITE_CACHE = {}


def codon_sites(codon):
    """Return the synonymous and non-synonymous site counts of one codon."""
    codon = codon.upper()
    if codon not in _SITE_CACHE:
        synonymous = _synonymous_site_count(codon)
        _SITE_CACHE[codon] = None if synonymous is None else (synonymous, 3 - synonymous)
    return _SITE_CACHE[codon]


def _count_differences(codon_a, codon_b):
    """
    Count synonymous and non-synonymous differences between two codons.

    When the codons differ at more than one position the order of the changes is
    unknown, so all possible orders are tried and the counts averaged, which is
    what the Nei and Gojobori method prescribes.
    """
    positions = [i for i in range(3) if codon_a[i] != codon_b[i]]
    if not positions:
        return 0.0, 0.0
    if len(positions) == 1:
        i = positions[0]
        same = sequences.translate_codon(codon_a) == sequences.translate_codon(codon_b)
        return (1.0, 0.0) if same else (0.0, 1.0)

    from itertools import permutations
    synonymous_total, nonsynonymous_total, paths = 0.0, 0.0, 0
    for route in permutations(positions):
        current = codon_a
        synonymous, nonsynonymous, broken = 0.0, 0.0, False
        for i in route:
            nxt = current[:i] + codon_b[i] + current[i + 1:]
            before, after = sequences.translate_codon(current), sequences.translate_codon(nxt)
            if before in ("X", "*") or after in ("X", "*"):
                broken = True
                break
            if before == after:
                synonymous += 1
            else:
                nonsynonymous += 1
            current = nxt
        if not broken:
            synonymous_total += synonymous
            nonsynonymous_total += nonsynonymous
            paths += 1
    if paths == 0:
        return 0.0, 0.0
    return synonymous_total / paths, nonsynonymous_total / paths


def _jukes_cantor(proportion):
    """
    Correct a raw difference rate for changes that happened more than once.

    Two sequences that have been apart a long time will have changed the same
    site twice, hiding one of the changes. This correction, from Jukes and
    Cantor, adjusts for that. It returns nothing when the sequences are too
    different for the correction to work.
    """
    if not np.isfinite(proportion) or proportion >= 0.75:
        return np.nan
    if proportion <= 0:
        return 0.0
    return float(-0.75 * np.log(1 - 4 * proportion / 3))


def consensus_codons(alignment):
    """
    Build the consensus sequence of a gene: the most common codon at every position.

    Every species is then compared to this consensus, which is a fast stand-in
    for comparing every species to its own ancestor.
    """
    length = min(len(s) for s in alignment.values())
    length -= length % 3
    consensus = []
    for start in range(0, length, 3):
        counts = Counter()
        for sequence in alignment.values():
            codon = sequence[start:start + 3].upper()
            if set(codon) <= set(_BASES):
                counts[codon] += 1
        consensus.append(counts.most_common(1)[0][0] if counts else "NNN")
    return consensus


def dnds_for_species(sequence, consensus):
    """
    Compare one species' gene to the consensus and return its dN, dS and omega.

    Codons that contain a gap, an N, or a stop are skipped.
    """
    synonymous_sites, nonsynonymous_sites = 0.0, 0.0
    synonymous_diff, nonsynonymous_diff = 0.0, 0.0

    for index, reference in enumerate(consensus):
        codon = sequence[index * 3:index * 3 + 3].upper()
        if not (set(codon) <= set(_BASES) and set(reference) <= set(_BASES)):
            continue
        sites_a, sites_b = codon_sites(codon), codon_sites(reference)
        if sites_a is None or sites_b is None:
            continue
        synonymous_sites += (sites_a[0] + sites_b[0]) / 2
        nonsynonymous_sites += (sites_a[1] + sites_b[1]) / 2
        s_diff, n_diff = _count_differences(codon, reference)
        synonymous_diff += s_diff
        nonsynonymous_diff += n_diff

    if synonymous_sites < 1 or nonsynonymous_sites < 1:
        return np.nan, np.nan, np.nan
    dn = _jukes_cantor(nonsynonymous_diff / nonsynonymous_sites)
    ds = _jukes_cantor(synonymous_diff / synonymous_sites)
    omega = dn / ds if (np.isfinite(dn) and np.isfinite(ds) and ds > 0) else np.nan
    return dn, ds, omega


def run(settings, progress=None):
    """
    Work out dN/dS for every gene in the alignment folder.

    Each species is compared to its gene's consensus sequence, and the per-gene
    row summarizes those species values. Returns one table with a row per gene
    and one with a row per species and gene.
    """
    say = progress or (lambda message: None)
    folder = Path(settings.alignment_folder)
    files = sorted([p for p in folder.iterdir()
                    if p.suffix.lower() in (".fasta", ".fa", ".fas", ".fna", ".aln")])

    per_gene, per_species = [], []
    for number, path in enumerate(files, start=1):
        gene = sequences.gene_name_from_file(path)
        say(f"Measuring selection on gene {number} of {len(files)}: {gene}")
        alignment = sequences.read_fasta(path)
        if len(alignment) < settings.dnds_min_species:
            continue
        consensus = consensus_codons(alignment)

        rows = []
        for species, sequence in alignment.items():
            dn, ds, omega = dnds_for_species(sequence, consensus)
            rows.append(dict(gene=gene, genome_id=species, dN=dn, dS=ds, omega=omega))
        per_species.extend(rows)

        frame = pd.DataFrame(rows)
        usable = frame[np.isfinite(frame.omega)]
        per_gene.append(dict(
            gene=gene,
            n_species=int(len(usable)),
            median_omega=float(usable.omega.median()) if len(usable) else np.nan,
            mean_dN=float(np.nanmean(frame.dN)),
            mean_dS=float(np.nanmean(frame.dS)),
            fraction_omega_above_1=float((usable.omega > 1).mean()) if len(usable) else np.nan,
        ))

    if not per_gene:
        raise ValueError(
            "No gene had enough species for a dN/dS estimate. Lower "
            "'dnds_min_species' in the settings, or check the alignment folder.")

    gene_table = pd.DataFrame(per_gene).sort_values("median_omega").reset_index(drop=True)
    gene_table["interpretation"] = np.where(
        gene_table.median_omega < 0.1, "strongly constrained",
        np.where(gene_table.median_omega < 0.5, "constrained",
                 np.where(gene_table.median_omega < 1, "weakly constrained", "possibly adaptive")))
    return gene_table, pd.DataFrame(per_species)
