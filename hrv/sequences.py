"""
Turning DNA alignments into a table of amino-acid variants.

Input : one FASTA file per gene, holding an in-frame codon alignment.
        Every sequence name is a genome assembly identifier, e.g. "hg38".
        The gene name is read from the file name (see `gene_name_from_file`).

Output: a genotype table with one row per species and one column per variant.
        1  = the species carries the alternative amino acid
        0  = the species carries the consensus amino acid
        NaN = the species has no amino acid called there (gap or ambiguous DNA)

Variant names look like "KCNQ1:K465R", which reads as: in gene KCNQ1, at
alignment position 465, the usual amino acid is K and the alternative is R.
"""

from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

# The standard genetic code, written out so no external package is needed.
_CODON_TABLE = {}
_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for _i, _b1 in enumerate(_BASES):
    for _j, _b2 in enumerate(_BASES):
        for _k, _b3 in enumerate(_BASES):
            _CODON_TABLE[_b1 + _b2 + _b3] = _AMINO[_i * 16 + _j * 4 + _k]


def translate_codon(codon):
    """
    Translate one three-letter DNA codon into a one-letter amino acid.

    Returns "X" if the codon contains a gap, an N, or any character that is
    not A, C, G or T. "X" always means "not called", never a real amino acid.
    """
    codon = codon.upper()
    return _CODON_TABLE.get(codon, "X")


def translate_sequence(dna):
    """
    Translate a whole aligned DNA sequence into a list of amino acids.

    The sequence is read three letters at a time from the start, because the
    alignments are in frame. Positions that cannot be translated become "X".
    """
    dna = str(dna).upper()
    return [translate_codon(dna[i:i + 3]) for i in range(0, len(dna) - 2, 3)]


def gene_name_from_file(path, field=1, separator="."):
    """
    Work out which gene a FASTA file holds, from its file name.

    The files provided are named like "ENST00000155840.KCNQ1.cleanLb_hmm_manual.fasta",
    so splitting on "." and taking field number 1 gives "KCNQ1".
    Change `field` or `separator` if your own files are named differently.
    """
    stem = Path(path).name
    for ending in (".fasta", ".fa", ".fas", ".fna", ".aln"):
        if stem.lower().endswith(ending):
            stem = stem[: -len(ending)]
            break
    parts = stem.split(separator)
    if field < len(parts):
        return parts[field]
    return parts[0]


def read_fasta(path):
    """
    Read a FASTA file into a dictionary of {sequence name: sequence text}.

    Only the first word of each ">" line is kept as the name, which is what
    the alignments use. Written out by hand so the repository needs no
    sequence-reading package.
    """
    sequences = {}
    name = None
    chunks = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        sequences[name] = "".join(chunks)
    return sequences


def protein_matrix_for_gene(fasta_path):
    """
    Read one gene alignment and translate it.

    Returns the list of species names, a table of amino acids with one row per
    species and one column per alignment position, and whether the sequences
    were of unequal length, which would mean the file is not a proper alignment.
    """
    sequences = read_fasta(fasta_path)
    names = list(sequences)
    proteins = [translate_sequence(sequences[n]) for n in names]
    if not proteins:
        return names, np.empty((0, 0), dtype="<U1"), False
    width = min(len(p) for p in proteins)
    ragged = len({len(p) for p in proteins}) > 1
    matrix = np.array([p[:width] for p in proteins])
    return names, matrix, ragged


def call_variants_in_gene(gene, names, proteins, all_species,
                          min_carriers=3, min_species=106):
    """
    Find the variable amino-acid positions of one gene.

    For every alignment position the most common amino acid across species is
    called the consensus. Every other amino acid seen at that position becomes
    its own variant, provided at least `min_carriers` species carry it and at
    least `min_species` species have any amino acid called there.

    Returns the genotype columns (one array per variant, in the order of
    `all_species`) and a description of each variant.
    """
    species_position = {s: i for i, s in enumerate(all_species)}
    row_of = np.array([species_position.get(n, -1) for n in names])

    columns, descriptions = [], []
    for position in range(proteins.shape[1]):
        letters = proteins[:, position]
        called = np.array([a not in ("X", "*", "-") for a in letters])
        counts = Counter(letters[called])
        if not counts:
            continue
        n_called = int(called.sum())
        if n_called < min_species:
            continue
        consensus = counts.most_common(1)[0][0]

        for alternative, n_carriers in counts.items():
            if alternative == consensus or n_carriers < min_carriers:
                continue
            genotype = np.full(len(all_species), np.nan)
            for row, is_called, letter in zip(row_of, called, letters):
                if row >= 0 and is_called:
                    genotype[row] = 1.0 if letter == alternative else 0.0
            columns.append(genotype)
            descriptions.append(dict(
                variant=f"{gene}:{consensus}{position + 1}{alternative}",
                gene=gene,
                position=position + 1,
                consensus_aa=consensus,
                alt_aa=alternative,
                n_carriers=int(n_carriers),
                n_species_called=n_called,
            ))
    return columns, descriptions


def build_variant_table(alignment_folder, all_species,
                        min_carriers=3, min_species=106,
                        gene_field=1, gene_separator=".", progress=None):
    """
    Turn a folder of gene alignments into the full genotype table.

    Goes through every FASTA file in the folder, translates it, calls the
    variants, and stacks the results into one wide table.

    Returns a genotype table (species by variant, values 1 / 0 / NaN) and a
    variant description table with one row per variant.
    """
    files = sorted([p for p in Path(alignment_folder).iterdir()
                    if p.suffix.lower() in (".fasta", ".fa", ".fas", ".fna", ".aln")])
    if not files:
        raise FileNotFoundError(f"No FASTA alignment files found in {alignment_folder}")

    all_columns, all_descriptions, unmatched = [], [], set()
    for number, path in enumerate(files, start=1):
        gene = gene_name_from_file(path, gene_field, gene_separator)
        if progress:
            progress(f"Reading gene {number} of {len(files)}: {gene}")
        names, proteins, ragged = protein_matrix_for_gene(path)
        if ragged and progress:
            progress(f"Warning: the sequences in {Path(path).name} are not all the "
                     f"same length, so the gene was cut to the shortest one "
                     f"({proteins.shape[1]} amino acids). Check the alignment.")
        unmatched |= {n for n in names if n not in set(all_species)}
        columns, descriptions = call_variants_in_gene(
            gene, names, proteins, all_species, min_carriers, min_species)
        all_columns.extend(columns)
        all_descriptions.extend(descriptions)

    if not all_columns:
        raise ValueError(
            "No variants passed the filters. Every position was either too "
            "conserved or too poorly sequenced. Try lowering 'fewest carriers' "
            "or 'fewest species sequenced' on the Data tab.")

    genotypes = pd.DataFrame(
        np.array(all_columns).T,
        index=list(all_species),
        columns=[d["variant"] for d in all_descriptions])
    variant_info = pd.DataFrame(all_descriptions)

    if unmatched and progress:
        progress(f"Note: {len(unmatched)} sequence names are not in the species "
                 f"file and were skipped: {', '.join(sorted(unmatched))}")
    return genotypes, variant_info
