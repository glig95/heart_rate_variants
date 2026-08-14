# hrv_variants

This repo contains code for identifying genetic variants linked to heart rate variability across mammals.
The program reads DNA alignments of a set of coding DNA sequences, finds which amino acid
each species carries at each position, constructs a matrix of all variants, and then searches for
variants that are associated with a faster or a slower heart rate.

Three methods for finding genetic variants are included: 1) phylogenetic regression as an established inference method in phylogenetics, 2) a new, rationally constructed clade-sharing method, and 3) a machine learning method called XGBoost. All three are explained in greater detail below.

Data needed to run the code is available in the repository. This includes the gene alignments, the
species table with heart rates and body masses, and the mammalian tree.

* **[The methods](#the-methods)**
* **[The data](#the-data-in-this-repository)**

---

## Installation

### Windows

For Windows there is a ready-built application on the
[Releases page](https://github.com/glig95/heart_rate_variants/releases) that
needs no Python at all.
Download `HeartRateVariants-windows.zip` from the
[Releases page](https://github.com/glig95/heart_rate_variants/releases), extract
it, and double-click `HeartRateVariants.exe`.

To run from the source instead, install Python 3.9 or newer from python.org with
**Add python.exe to PATH** ticked, then double-click `run_windows.bat`.

### macOS

Needs a Python that can open a window, which the one built into macOS cannot do.
Install it from [python.org](https://www.python.org/downloads/macos/) or with
`brew install python-tk`, then:

```bash
git clone https://github.com/glig95/heart_rate_variants.git
cd heart_rate_variants

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python run_gui.py
```

### Linux

```bash
sudo apt install python3 python3-venv python3-tk      # Debian, Ubuntu, Mint
# sudo dnf install python3 python3-tkinter            # Fedora, RHEL
# sudo pacman -S python tk                            # Arch

git clone https://github.com/glig95/heart_rate_variants.git
cd heart_rate_variants

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python run_gui.py
```

---

## Running the program

After running the GUI, a window with three tabs will open.
First, check the data setting tab, then the method setting tab, and finally press **Run the analysis** button.
A run should take about thirty seconds; XGBoost takes a few minutes, because it searches for its own optimal settings.

Tabs:

**Data settings.** Where the input files are, and the two rules that decide which
positions in a protein count as a variant: how many species must carry it, and
how many must have been sequenced there. Also which tree to use, and what to do with
species that have no amino acid identified.

**Method settings.** Which of the four methods to run, whether body mass is
treated as a confounder, how clades are defined, and what counts as a result. The
Run button is here.

**Results.** Fills in after a run: variants are summarized in a table, and
figures. Clicking a row opens that variant's figure. 
The **share explained** and **p-value** column headings carry a question
mark; clicking either opens a page explaining that number.

The log along the bottom fills up while a run is going, and **Stop** interrups it.

---

## The methods

PGLS and the clade-sharing method are statistical tests, which test one variant at
a time, and assign a p-value to each. XGBoost is a machine learning model that ranks
variants according to how much they were used to inform prediction, rather than the p-value.
A fourth option on the Method settings tab runs PGLS and the clade-sharing method together,
keeping only the variants both select as significant.

The three methods answer slightly different questions. PGLS asks whether carriers have
different heart rates across the whole tree, once shared ancestry is accounted for. The clade-sharing
method asks a similar question in a modified different way, that does not require the tree:
whether the variant has the same effect repeatedly, in separate
groups of relatives. XGBoost asks whether the variant helps predict heart rate
when every other variant is available too, however, it has no knowledge about phylogeny.

### PGLS, phylogenetic generalized least squares

PGLS is similar to linear regression, but adapted to evolutionary data. The usual
regression treats observations as independent. However, species that share ancestry
are not independent datapoints. PGLS keeps the ordinary regression form but changes the
assumption about the distribution of residuals. The model is

```
y  =  a + b x + e ,        e  ~  Normal(0, s² V)
```

where `y` is log10 resting heart rate across species, `x` is the genotype (1 for
a carrier, 0 for a non-carrier) and `V` is a covariance matrix read off the tree.
`V[i][j]` is the length of the path from the root that species *i* and *j* share,
which is the distance from the root to their most recent common ancestor;
`V[i][i]` is the whole root-to-tip length.

Writing `V = L Lᵀ` and multiplying the whole model through by the inverse of `L`
leaves residuals that are independent and equally variable, so ordinary least
squares on the transformed data is the generalized least-squares fit of the
original.

**Output.** The change in heart rate in carriers as a percentage; 
the squared phylogeny-corrected partial correlation, which is the
share of the variation in heart rate the variant accounts for; and a p-value from
the t-test.

**References.**

* Felsenstein, J. (1985). Phylogenies and the comparative method. *The American
  Naturalist* 125(1), 1–15. [doi:10.1086/284325](https://doi.org/10.1086/284325)
  — the paper that made the problem, and the phylogenetic contrast, standard.
* Grafen, A. (1989). The phylogenetic regression. *Philosophical Transactions of
  the Royal Society of London B* 326(1233), 119–157.
  [doi:10.1098/rstb.1989.0106](https://doi.org/10.1098/rstb.1989.0106) — where
  the regression-with-a-tree-covariance form used here was introduced.
* Freckleton, R. P., Harvey, P. H. and Pagel, M. (2002). Phylogenetic analysis
  and comparative data: a test and review of evidence. *The American Naturalist*
  160(6), 712–726. [doi:10.1086/343873](https://doi.org/10.1086/343873) — the
  standard practical reference for PGLS as it is used today, and the one to cite
  if only one is wanted.

A run of this pipeline was checked against a textbook generalized least-squares fit and
agreed with it.

### The clade-sharing method

The clade-sharing method is developed specifically for this dataset, inspired by the
observation that a variant can have the same effect across different clades
of animals.
Specifically, it systematically searches for variants that are associated with
a change in heart rate, in lineages that acquired it separately.

**How is this different from PGLS.** It is similar, but unlike PGLS, it does not
depend on the tree. It is formulated to find variants that satisfy the following criteria,
which can be intuitively grasped and manually checked, which is harder for PGLS:

1. Is the variant present in only some species of a clade, and in those species (carriers)
   it causes a shift in heart rate compared to non-carriers?
2. Does it have the same effect in several clades?

**Detailed description of clade-sharing**

1. Every species is assigned to a clade. By default the clade is the taxonomic
   order from the species file, meaning, *Rodentia*, *Chiroptera*, *Primates* and so
   on. Alternatively, the tree can be cut at a chosen depth, and each piece below
   the cut becomes a clade.
2. For one variant, each clade is inspected. A clade that holds both carriers and
   non-carriers is a **contrast clade**: it contains a comparison that can be
   made between close relatives. A clade where every species carries the variant,
   or none does, holds no comparison and takes no further part in that variant's
   test.
3. Inside each contrast clade, that clade's own mean heart rate and its own mean
   genotype are subtracted from its species. What is left is how much faster or
   slower than its clademates each species beats, and whether it carries the
   variant when its clademates mostly do not.
4. Those centered values are pooled across all contrast clades and the
   correlation between them is calculated. That correlation is the statistic, and
   it is tested against zero with a t-test.

In symbols, with `c(i)` the clade of species *i*:

```
y*[i]  =  y[i] − mean of y over clade c(i)
x*[i]  =  x[i] − mean of x over clade c(i)
r      =  correlation( y* , x* )   over the species in contrast clades
```

**Summary of the idea behind clade-sharing.** Subtracting each clade's mean removes
the shared ancestry of animals within the clade. However, this means that
two sister species in the same order are treated as two independent comparisons
even though they are not fully independent. PGLS does not make this assumption
but requires a tree, pointing to trade-offs that have to be made. This is the main reason to
run PGLS alongside clade-sharing to verify how much they correlate.

**Input: the number of contrast clades required.** This is set on the Method settings
tab and the default is one, which means any variant with at least one usable
within-clade comparison is tested.

**Output.** The within-clade correlation and its p-value; the squared
correlation as the share explained; the percentage change in heart rate in
carriers; how many contrast clades there were; and which clades those were, with
the carrier count in each.

### PGLS and clade-sharing run simultaneously

This option finds the intersection of the two methods above: a variant is kept only
when PGLS and the clade-sharing method both find it scores over the p-value threshold.
The p-value shown for a variant is the weaker of its two (one from PGLS and
one from clade-sharing). The run also produces `methods_compared.png`, which
plots the two effect sizes against each other and the two strengths of evidence
against each other.

### XGBoost

**Rationale.** XGBoost serves as an initial test to see whether heart rate is at
all predictable given the genomic data. It fits many decision tree models from all variants
at once, keeps the most accurate, and lists the variants that the model made most
use of. It does not account for the phylogeny at all, so a variant can score highly
simply because it marks a group of relatives who happen to share a heart rate.
It outputs variant's importance rather than a p-value: importance says how much a
variant improved the model's fit.

**Input.** The algorithm performs grid search over hyperparameters, so there
is no user input required here. Varied parameters are: the number of trees, tree depth,
learning rate, what fraction of the species each tree sees, and how strongly large
weights are penalized.

**Scoring.** The winner is scored two ways: on a random split of species, and
leaving out one whole clade at a time. The random split generally scores higher,
because a test species usually has a close relative sitting in the training set
and the model can lean on it. The leave-one-clade-out score is constructed as a more
fair evaluation.

**Output.** Each variant's share of the model's total importance, and the
fall in leave-one-clade-out accuracy when that variant is removed, and the model
refitted.

**Reference.**

* Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system.
  *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
  Discovery and Data Mining*, 785–794.
  [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

---

## Body mass as a confounder

Body size is the strongest predictor of heart rate in mammals: on this data it
explains about three quarters of the variation, with heart rate scaling as
mass to the power of about -0.19. A variant that is simply commoner in small
animals will therefore look associated with a fast heart for that reason alone.

Ticking **Treat body mass as a confounder** removes the body-mass effect from
heart rate before any variant is tested. The residuals of heart rate are calculated
by performing a linear fit in log-log coordinates, and what each species has left
over is used as a normalized heart rate variability and as an input to the three predictors.

Body mass is read from the `body_mass_g` column of `data/species_info.csv`, the
same file the heart rates and the species names come from.

One quantity was estimated from the data to make that adjustment, so one degree
of freedom is subtracted in both statistical methods.

Species with no body mass in the species file take no part in these runs, and the
run notes say which. In the supplied data that is one species.

---

## How are clades defined

There are four different ways in which this can be achieved, which are summarized below.

| Rule | What it does |
|---|---|
| Taxonomic orders | Rodentia, Carnivora, Chiroptera and so on, from the species file. Defined outside the data being tested, which is what makes them fair. This is the default. |
| Taxonomic orders, small ones merged | The same, but any order with fewer than a set number of species joins the order whose median heart rate is closest. Note that the merging uses heart rate, the trait being tested, which makes it a less conservative choice. |
| Cut the tree into a set number of groups | The whole tree starts as one group and the largest group is opened up next, until there are enough. Every group is one whole branch. The number is a free choice; more, smaller groups make a stricter test. |
| My own file | A `.csv` with columns `genome_id` and `clade`. Species not listed take no part in the clade-sharing test. |

---

## Species with no amino acid called

A species has no call at a position when the alignment has a gap there, the DNA
contains an N, or the codon cannot be translated. About one cell in ten is like
this. A zero is not a missing value: it means the species was sequenced and
carries the usual amino acid.

Three rules are offered.

**Copy from the closest species.** This is the default. Every pair of species is
compared on the fraction of positions at which they differ, across all genes. An
uncalled cell takes the value of the closest species that has a call there; if
that species has none either, the search moves to the next closest, and so on.
Every filled cell is then a real observation from a close relative rather than an
average. A cell stays uncalled only if no species at all has a call for that
variant.

**Leave the species out of that test.** The species is dropped from the test of
that one variant. Nothing is invented, but each variant is then tested on a
slightly different set of species, and the phylogenetic correction has to be
redone for each, which is slower.

**Fill with the variant's average.** Each variant is a column of ones and zeros
with gaps. The average of the called values is the fraction of sequenced species
carrying the variant, and every gap is replaced by that fraction. A species with
no call is therefore treated as neither a carrier nor a clear non-carrier, and
the number of species stays the same for every variant.

One should be careful when using the last option, as it might cause false correlations.

The choice among the options above governs PGLS.
The other methods have their own fixed rule: the clade-sharing method always leaves an
uncalled species out of the variant concerned, and the machine learning model (XGBoost)
always fills within each training fold, so that nothing leaks from the held-out species.

How much is actually missing, in total and per species and per gene, is written
to `missing_variants_per_species.csv` and `missing_variants_per_gene.csv` by every
run, and drawn in `plots/missing_variants.png`.

---

## Output

Each run writes a dated folder under `results/`:

| File | What is in it |
|---|---|
| `variants_found.csv` | The variants that passed, numbered, strongest first |
| `all_variants_tested.csv` | Every variant, whether it passed or not |
| `summary_per_species.csv` | Per species: which of them it carries |
| `summary_per_gene.csv` | Per gene: how many tested, how many passed, the strongest |
| `carriers_per_variant.csv` | One row per variant per species |
| `cooccurrence.csv` | Which variants the same species carry: one row per set, with how many species carry it, how many variants are in it, and how many clades those species span |
| `clades_used.csv` | The clades the run actually used, with their sizes |
| `missing_variants_per_gene.csv` | Per gene: how many variants were missing, meaning no amino acid was called there, how many the missing-data rule gave a value to, and how many of those were written in as present |
| `missing_variants_per_species.csv` | The same counts, per species |
| `all_models_tried.csv` | XGBoost only: every parameter combination and its score |
| `model_accuracy.csv` | XGBoost only: how accurate the best model was |
| `settings_used.json` | Every setting, so the run can be repeated |
| `run_notes.txt` | The clade rule in words, plus notes about the data |
| `plots/` | The figures |

---

## The data in this repository

Input needed to run the pipeline is in `data/`.

| Folder or file | What it holds |
|---|---|
| `data/alignments/` | 19 FASTA files, one per cardiac ion-channel gene, holding in-frame codon alignments across up to 153 mammals. Sequence names are genome assembly identifiers, `hg38` for human and `mm39` for mouse. Together, 8.7 MB |
| `data/species_info.csv` | 153 rows: genome identifier, scientific and common name, resting heart rate in beats per minute, adult body mass in grams, and order, family and genus |
| `data/trees/ebisuya_tree.nwk` | The tree the program uses, covering all 153 species |
| `data/trees/other_trees/` | Further trees kept for comparison, none of them used unless you browse to one: a dated tree pruned from Upham et al., the same phylogeny with unresolved dating, a tree built from the gene sequences themselves, and the table saying how each species was matched to a tip |
| `data/README_data.md` | Every column and every file, described in full, including which species is missing a body mass and which pair the tree cannot separate |

---

## References

* Felsenstein, J. (1985). Phylogenies and the comparative method. *The American
  Naturalist* 125(1), 1–15. [doi:10.1086/284325](https://doi.org/10.1086/284325)
* Grafen, A. (1989). The phylogenetic regression. *Philosophical Transactions of
  the Royal Society of London B* 326(1233), 119–157.
  [doi:10.1098/rstb.1989.0106](https://doi.org/10.1098/rstb.1989.0106)
* Freckleton, R. P., Harvey, P. H. and Pagel, M. (2002). Phylogenetic analysis
  and comparative data: a test and review of evidence. *The American Naturalist*
  160(6), 712–726. [doi:10.1086/343873](https://doi.org/10.1086/343873)
* Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system.
  *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
  Discovery and Data Mining*, 785–794.
  [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

The clade-sharing method has no reference of its own, because it is not a
published named method. It is a within-group regression with clade as the
grouping factor, which is standard, applied to the phylogenetic problem in the
way described above.
