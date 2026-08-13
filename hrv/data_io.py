"""
Loading the input files, lining them up, and dealing with uncalled genotypes.

Three things have to agree before any analysis can run: the species table
(heart rate, body mass, taxonomy), the genotype table built from the
alignments, and the phylogenetic tree. This file loads them, keeps only the
species all of them share, and hands back one Dataset object.

Building the genotype table from the alignments takes several seconds, so the
result is cached. The cache is keyed on the alignment files themselves and on
the two rules that decide which positions become variants, so any change to
either forces a fresh read.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from . import sequences, trees
from .config import CACHE_FOLDER


# The variant table for the most recent set of inputs, kept for as long as the
# program is open. Repeated runs in one session then skip the disk entirely.
_IN_MEMORY = {}


@dataclass
class Dataset:
    """Everything one analysis needs, with all parts already lined up."""

    species: list                # genome identifiers, the row order everywhere
    species_table: pd.DataFrame  # names, heart rate, body mass, taxonomy
    genotypes: pd.DataFrame      # species by variant, values 1 / 0 / NaN
    variant_info: pd.DataFrame   # one row per variant
    log_heart_rate: np.ndarray   # log10 of resting heart rate, in bpm
    log_mass: np.ndarray         # log10 of body mass, in grams (NaN if unknown)
    covariance: pd.DataFrame     # shared ancestry between species, from the tree
    notes: list                  # anything the user should know about the load

    def scientific_names(self):
        """Return the Latin name of every species, in the dataset's row order."""
        lookup = dict(zip(self.species_table.genome_id, self.species_table.scientific_name))
        return [lookup.get(s, s) for s in self.species]


def load_species_table(path):
    """
    Read the species file.

    Required columns: genome_id, scientific_name, heart_rate_bpm.
    Optional but used when present: common_name, body_mass_g, order, family, genus.
    Species without a heart rate are dropped, since heart rate is what the whole
    pipeline predicts.
    """
    table = pd.read_csv(path)
    needed = ["genome_id", "scientific_name", "heart_rate_bpm"]
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise ValueError(f"The species file is missing these columns: {missing}")

    for column, filler in [("common_name", ""), ("body_mass_g", np.nan),
                           ("order", "unknown"), ("family", "unknown"), ("genus", "unknown")]:
        if column not in table.columns:
            table[column] = filler

    table["genome_id"] = table.genome_id.astype(str)
    table = table[table.heart_rate_bpm.notna()].reset_index(drop=True)
    return table


def _cache_key(settings, species):
    """
    Build a short identifier for one particular variant table.

    It covers the name, size and modification time of every alignment file, the
    two calling filters, and the list of species. Change any of those and the
    key changes, so a stale cache can never be used by mistake.
    """
    parts = [str(settings.min_carriers), str(settings.min_species_per_variant)]
    folder = Path(settings.alignment_folder)
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() in (".fasta", ".fa", ".fas", ".fna", ".aln"):
            info = path.stat()
            parts.append(f"{path.name}:{info.st_size}:{int(info.st_mtime)}")
    parts.append(",".join(species))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _read_cache(key):
    """Return the cached variant table for this key, or None if there is none."""
    folder = CACHE_FOLDER / key
    if not (folder / "genotypes.npz").exists():
        return None
    try:
        stored = np.load(folder / "genotypes.npz", allow_pickle=True)
        genotypes = pd.DataFrame(stored["values"],
                                 index=[str(s) for s in stored["species"]],
                                 columns=[str(c) for c in stored["variants"]])
        return genotypes, pd.read_csv(folder / "variant_info.csv")
    except Exception:
        return None


def _write_cache(key, genotypes, variant_info):
    """Save a variant table so the next run with the same inputs can skip the work."""
    folder = CACHE_FOLDER / key
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "genotypes.npz",
                        values=genotypes.values.astype(np.float32),
                        species=np.array(list(genotypes.index)),
                        variants=np.array(list(genotypes.columns)))
    variant_info.to_csv(folder / "variant_info.csv", index=False)
    (folder / "what_this_is.txt").write_text(
        "A saved copy of the variant table, so runs after the first do not have "
        "to read the alignments again. Deleting this folder is always safe.\n")


def load_dataset(settings, progress=None):
    """
    Load everything the pipeline needs and line it up.

    Reads the species table, builds the variant table from the alignments (or
    takes it from the cache), reads the tree, then keeps only the species that
    appear in all three. Anything dropped along the way is recorded in `notes`.
    """
    notes = []
    say = progress or (lambda message: None)

    say("Reading the species file")
    species_table = load_species_table(settings.species_file)
    species = list(species_table.genome_id)

    key = _cache_key(settings, species) if settings.use_cache else None
    cached = _IN_MEMORY.get(key) if key else None
    where = "memory"
    if cached is None and key:
        cached = _read_cache(key)
        where = "cache folder"

    if cached is not None:
        genotypes, variant_info = cached[0].copy(), cached[1].copy()
        say(f"Reusing the prepared variant table from {where} "
            f"({genotypes.shape[1]} variants)")
        notes.append(f"The variant table was reused from {where}. It is rebuilt "
                     f"automatically whenever the alignments or the calling filters "
                     f"change.")
        _IN_MEMORY.clear()
        _IN_MEMORY[key] = (genotypes.copy(), variant_info.copy())
    else:
        say("Building the variant table from the alignments")
        genotypes, variant_info = sequences.build_variant_table(
            settings.alignment_folder, species,
            min_carriers=settings.min_carriers,
            min_species=settings.min_species_per_variant,
            progress=say)
        if key:
            _write_cache(key, genotypes, variant_info)
            _IN_MEMORY.clear()          # only ever hold one, to stay small
            _IN_MEMORY[key] = (genotypes.copy(), variant_info.copy())

    notes.append(f"{genotypes.shape[1]} variants called in "
                 f"{variant_info.gene.nunique()} genes.")

    say("Reading the tree")
    covariance = trees.load_tree_covariance(settings.tree_file, notes)

    shared = [s for s in genotypes.index if s in covariance.index]
    dropped = [s for s in genotypes.index if s not in covariance.index]
    if dropped:
        notes.append(f"{len(dropped)} species are not in the tree and were left "
                     f"out: {', '.join(dropped)}")

    species_table = species_table[species_table.genome_id.isin(shared)]
    species_table = species_table.set_index("genome_id").loc[shared].reset_index()
    genotypes = genotypes.loc[shared]
    covariance = covariance.loc[shared, shared]

    # Species have just been dropped, so carriers are recounted and the filters
    # reapplied. Otherwise a variant could be listed with more carriers than
    # the analysis actually saw.
    genotypes, variant_info, removed = recount_and_refilter(
        genotypes, variant_info, settings.min_carriers, settings.min_species_per_variant)
    if removed:
        notes.append(f"{removed} variants no longer met the filters once species "
                     f"missing from the tree were dropped, and were removed.")

    # Species sitting at exactly the same point on the tree, which happens when a
    # branch has length zero, are worth knowing about. The analysis still runs.
    values = covariance.values
    off_diagonal_max = (values - np.diag(np.diag(values))).max(axis=1)
    identical = int(np.sum(np.isclose(np.diag(values), off_diagonal_max)))
    if identical:
        notes.append(f"{identical} species sit at the same point on the tree as "
                     f"another species, because their branch has length zero. They "
                     f"are kept, and the phylogenetic correction cannot tell them apart.")

    log_heart_rate = np.log10(species_table.heart_rate_bpm.values.astype(float))
    mass = species_table.body_mass_g.values.astype(float)
    log_mass = np.where(np.isfinite(mass) & (mass > 0),
                        np.log10(np.where(mass > 0, mass, 1)), np.nan)
    if np.isnan(log_mass).any():
        unknown = species_table.scientific_name[np.isnan(log_mass)].tolist()
        notes.append(f"{len(unknown)} species have no body mass and are left out "
                     f"of runs that treat body mass as a confounder: {', '.join(unknown)}")

    uncalled = float(np.mean(~np.isfinite(genotypes.values)))
    notes.append(f"{100 * uncalled:.1f}% of the genotype table is uncalled.")
    notes.append(f"{len(shared)} species used in total.")
    say(f"Ready: {len(shared)} species, {genotypes.shape[1]} variants")

    return Dataset(species=shared, species_table=species_table, genotypes=genotypes,
                   variant_info=variant_info, log_heart_rate=log_heart_rate,
                   log_mass=log_mass, covariance=covariance, notes=notes)


def recount_and_refilter(genotypes, variant_info, min_carriers, min_species):
    """
    Recount carriers on the species actually being analyzed, and drop any
    variant that no longer meets the filters.

    Needed because variants are first called on whatever is in the alignment
    files, and species can be dropped afterwards for having no heart rate or for
    being absent from the tree. Without this step the carrier counts in the
    results would describe species that took no part in the test.
    """
    values = genotypes.values.astype(float)
    n_carriers = np.nansum(values == 1, axis=0).astype(int)
    n_called = np.sum(np.isfinite(values), axis=0).astype(int)

    keep = (n_carriers >= min_carriers) & (n_called >= min_species)
    removed = int((~keep).sum())

    variant_info = variant_info.copy()
    variant_info["n_carriers"] = n_carriers
    variant_info["n_species_called"] = n_called
    variant_info = variant_info[keep].reset_index(drop=True)
    genotypes = genotypes.loc[:, keep]
    return genotypes, variant_info, removed


def restrict_to_species_with_mass(dataset):
    """
    Return a copy of the dataset holding only the species that have a body mass.

    Used when body mass is treated as a confounder, so that every table, plot
    and summary from that run describes exactly the species the test used.
    """
    keep = np.isfinite(dataset.log_mass)
    if keep.all():
        return dataset
    species = [s for s, ok in zip(dataset.species, keep) if ok]
    genotypes = dataset.genotypes.loc[species]
    genotypes, variant_info, _ = recount_and_refilter(genotypes, dataset.variant_info, 1, 1)
    return Dataset(
        species=species,
        species_table=dataset.species_table[keep.tolist()].reset_index(drop=True),
        genotypes=genotypes,
        variant_info=variant_info,
        log_heart_rate=dataset.log_heart_rate[keep],
        log_mass=dataset.log_mass[keep],
        covariance=dataset.covariance.loc[species, species],
        notes=list(dataset.notes))


def species_distance_matrix(genotypes):
    """
    Fraction of called positions at which each pair of species differs.

    Only positions where both species have a call are counted, so a pair that
    shares little sequence is compared on what they do share rather than being
    treated as identical. Used to find each species' nearest neighbors.
    """
    values = np.asarray(genotypes, dtype=float)
    known = np.isfinite(values)
    present = known.astype(float)
    carried = np.where(known, values, 0.0)
    absent = np.where(known, 1 - values, 0.0)

    both_known = present @ present.T
    agreeing = carried @ carried.T + absent @ absent.T
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = 1 - agreeing / both_known
    finite = np.isfinite(distance)
    distance[~finite] = 1.0 if not finite.any() else float(np.nanmax(distance[finite]))
    np.fill_diagonal(distance, 0.0)
    return (distance + distance.T) / 2


def fill_from_nearest_species(genotypes, progress=None):
    """
    Fill every uncalled genotype with the value of the most similar species.

    For each species the others are sorted by how different their sequences are,
    measured over all genes. An uncalled cell takes the value of the closest
    species that has a call there; if that species has none either, the search
    moves on to the next closest, and so on. A cell stays uncalled only if no
    species at all has a call for that variant.

    Returns the filled matrix and how many cells were filled.
    """
    say = progress or (lambda message: None)
    values = np.asarray(genotypes, dtype=float).copy()
    n_species = values.shape[0]

    say("Comparing every pair of species to find the closest ones")
    distance = species_distance_matrix(values)
    neighbor_order = np.argsort(distance, axis=1)

    filled = 0
    for i in range(n_species):
        gaps = np.where(~np.isfinite(values[i]))[0]
        if gaps.size == 0:
            continue
        say(f"Filling {gaps.size} uncalled positions for species {i + 1} of {n_species}")
        remaining = gaps
        for neighbor in neighbor_order[i]:
            if neighbor == i or remaining.size == 0:
                continue
            candidate = values[neighbor, remaining]
            usable = np.isfinite(candidate)
            if usable.any():
                values[i, remaining[usable]] = candidate[usable]
                filled += int(usable.sum())
                remaining = remaining[~usable]
    return values, filled


def genotypes_for_analysis(dataset, settings, progress=None):
    """
    Return the genotype matrix in the form the chosen missing-data rule requires.

    "average": uncalled cells filled with the variant's column average.
    "nearest": uncalled cells filled from the closest species that has a call.
    "ignore":  uncalled cells left as they are, for each test to drop.
    """
    values = dataset.genotypes.values.astype(float)
    if settings.missing_data == "average":
        column_means = np.nanmean(values, axis=0)
        return np.where(np.isnan(values), column_means, values)
    if settings.missing_data == "nearest":
        filled, _ = fill_from_nearest_species(values, progress)
        return filled
    return values


def remove_body_mass_effect(dataset, progress=None):
    """
    Take the body-mass effect out of heart rate, once, for every method to share.

    Small animals have fast hearts, so a variant that is simply commoner in small
    animals looks associated with heart rate for that reason alone. This fits
    log10 heart rate against log10 body mass and keeps what is left over. Every
    method then asks the same question: does this variant go with a heart beating
    faster or slower than the animal's size would predict?

    The fit is a plain least-squares line through the species, with no phylogeny
    in it. The tree is used later, by whichever method needs it, but not here:
    this step is only meant to take size out of heart rate, and bringing the tree
    into it would change the slope on the basis of assumptions about how heart
    rate evolved.

    Body mass is read from the `body_mass_g` column of the species file, the same
    file the heart rates and the names come from. Returns a new dataset; the
    original is untouched.
    """
    say = progress or (lambda message: None)
    mass = dataset.log_mass
    heart_rate = dataset.log_heart_rate
    if not np.isfinite(mass).all():
        missing = int((~np.isfinite(mass)).sum())
        raise ValueError(f"{missing} species have no body mass, so it cannot be removed "
                         f"from heart rate. Fill in the body_mass_g column or turn the "
                         f"body-mass setting off.")

    design = np.column_stack([np.ones(len(mass)), mass])
    coefficients, *_ = np.linalg.lstsq(design, heart_rate, rcond=None)
    residual = heart_rate - design @ coefficients

    slope = float(coefficients[1])
    say(f"Removed the body-mass effect from heart rate: heart rate scales as "
        f"mass^{slope:.3f}")
    notes = list(dataset.notes)
    notes.append(
        f"Body mass was removed from heart rate before any variant was tested. Across "
        f"these species heart rate scales as mass^{slope:.3f}, fitted as a plain "
        f"least-squares line with no phylogeny in it, and every method was given what "
        f"is left after that. A result therefore means the variant goes with a heart "
        f"beating faster or slower than the animal's size predicts, not simply that "
        f"its carriers are small or large.")

    return Dataset(
        species=list(dataset.species),
        species_table=dataset.species_table,
        genotypes=dataset.genotypes,
        variant_info=dataset.variant_info,
        log_heart_rate=residual,
        log_mass=dataset.log_mass,
        covariance=dataset.covariance,
        notes=notes)
