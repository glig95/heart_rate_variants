"""
Settings for the whole pipeline.

Everything the user can choose lives in one place: the Settings object below.
The window fills this object in, and every analysis function reads from it.
Nothing else in the code stores options.

All default paths are worked out from the location of this file, so the folder
can be moved or renamed and everything still points at the right place.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import sys


# Where everything lives.
#
# Run from the source folder, this file is in <repository>/hrv/, so the
# repository is one level up. Run from a packaged application, there is no
# source folder: the data sits beside the executable instead, which is also what
# lets someone swap in their own data without rebuilding anything.
CODE_FOLDER = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    REPO_FOLDER = Path(sys.executable).resolve().parent
else:
    REPO_FOLDER = CODE_FOLDER.parent
DATA_FOLDER = REPO_FOLDER / "data"
CACHE_FOLDER = REPO_FOLDER / "cache"


@dataclass
class Settings:
    """
    All choices for one run of the pipeline.

    Every field has a default, so `Settings()` on its own is a complete,
    working configuration.
    """

    # ---------------------------------------------------------------- input
    alignment_folder: str = str(DATA_FOLDER / "alignments")
    species_file: str = str(DATA_FOLDER / "species_info.csv")
    tree_file: str = str(DATA_FOLDER / "trees" / "ebisuya_tree.nwk")
    output_folder: str = str(REPO_FOLDER / "results")

    # ------------------------------------------------- building the variants
    # A position in a protein becomes a variant only if enough species carry
    # the alternative amino acid and enough species are sequenced there.
    min_carriers: int = 2
    min_species_per_variant: int = 100

    # What to do with species that have no amino acid called at a position.
    # "nearest" copies the value of the most similar species that has one.
    # "ignore"  drops those species from that variant's test only.
    # "average" fills with the variant's own average across called species.
    missing_data: str = "nearest"

    # Reuse the variant table between runs when nothing about it has changed.
    use_cache: bool = True

    # ---------------------------------------------------------- which method
    # One of: "pgls", "clade_sharing", "combined", "xgboost"
    method: str = "combined"

    # Should body mass be treated as a confounder in this run?
    use_mass: bool = False

    # --------------------------------------------------- how clades are made
    # One of: "order", "merged_orders", "tree_cut", "custom_file"
    clade_definition: str = "order"
    n_tree_clades: int = 25            # only used by "tree_cut"
    min_clade_size: int = 3            # only used by "merged_orders"
    custom_clade_file: str = ""        # only used by "custom_file"

    # --------------------------------------------------------- what counts
    p_threshold: float = 1e-3          # a variant is kept below this p-value
    # Two by default: a variant seen in only one clade may be a single
    # evolutionary event rather than a pattern that has recurred.
    min_contrast_clades: int = 2       # clade-sharing: clades needed to accept a hit

    # ------------------------------------------------------------- XGBoost
    # The parameters are searched automatically, so they are not asked for in
    # the window. These two control the size of the search and the output.
    xgb_search_size: int = 150         # number of parameter combinations tried
    xgb_n_report: int = 20             # how many top variants to describe
    xgb_random_folds: int = 5

    # -------------------------------------------------------------- output
    random_seed: int = 0

    # ----------------------------------------------- exploring the data only
    dnds_min_species: int = 20         # genes with fewer species are skipped
    pca_n_axes: int = 6

    def save(self, path):
        """Write these settings to a .json file so a run can be repeated later."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path):
        """Read settings back from a .json file written by `save`."""
        stored = json.loads(Path(path).read_text())
        known = {f for f in Settings.__dataclass_fields__}
        return Settings(**{k: v for k, v in stored.items() if k in known})

    def describe_clades(self):
        """
        Return a short description, in plain words, of how clades will be built
        with the current choices. The window shows this text to the user.
        """
        if self.clade_definition == "order":
            return (
                "Clades are taxonomic orders (Rodentia, Carnivora, Chiroptera and "
                "so on), taken from the order column of the species file.\n"
                "An order counts for a variant only if it contains both carriers "
                "and non-carriers of that variant. An order with a single sampled "
                "species can never do that, so it is skipped."
            )
        if self.clade_definition == "merged_orders":
            return (
                f"Clades are taxonomic orders, except that every order with fewer "
                f"than {self.min_clade_size} species is merged into the order whose "
                "median heart rate is closest to its own.\n"
                "Note that the merging uses heart rate, the trait being tested, "
                "which makes it a less conservative choice than plain orders."
            )
        if self.clade_definition == "tree_cut":
            return (
                f"Clades are about {self.n_tree_clades} groups cut out of the "
                "phylogenetic tree. The whole tree starts as one group, and "
                "whichever group holds the most species is opened up next, until "
                "there are enough groups.\n"
                "Every group is one whole branch of the tree, so the groups are "
                "monophyletic. A branch that splits three or more ways is opened "
                "all at once, so the final count can come out slightly above the "
                "number requested. The groups actually used are written to "
                "clades_used.csv with every run."
            )
        return (
            "Clades are read from your own file.\n"
            "The file must be a .csv with two columns: genome_id and clade. Every "
            "species you want tested needs a row; species that are not listed take "
            "no part in the clade-sharing test."
        )

    def describe_missing_data(self):
        """Return a description, in plain words, of the missing-data rule in force."""
        if self.missing_data == "average":
            return (
                "Every uncalled cell is replaced by the average of that variant's "
                "column across the species that do have a call. Because the column "
                "holds ones and zeros, that average is the fraction of sequenced "
                "species carrying the variant, so an uncalled species is treated as "
                "neither a carrier nor a clear non-carrier.\n"
                "Take care with this one. Every species then counts toward every "
                "variant, including species that were never sequenced at that "
                "position, so each test looks better supported than it is. That can "
                "push a variant past the threshold on the strength of invented "
                "values, and it pulls effect sizes toward zero. It is the fastest "
                "option and it keeps the number of species the same for every "
                "variant, but a variant that is significant only under this rule "
                "should be checked under one of the others before it is believed."
            )
        if self.missing_data == "ignore":
            return (
                "A species with no call is left out of the test of that variant, "
                "and only of that variant. Nothing is invented, but each variant is "
                "then tested on a slightly different set of species."
            )
        return (
            "For each uncalled cell, the most similar species that does have a call "
            "there is found, and its value is copied across. Similarity is the "
            "fraction of amino-acid positions at which two species differ, measured "
            "over all genes. Species are tried in order of similarity until one with "
            "a call is found."
        )
