# The data that ships with this repository

Everything here can be replaced with your own files; see the "Using your own
data" section of the main README.

## `alignments/`

Nineteen FASTA files, one per cardiac ion-channel gene, holding in-frame DNA
codon alignments across up to 153 mammals. Sequence names are genome assembly
identifiers, for example `hg38` for human and `mm39` for mouse. The gene name is
the part of the file name between the first and second dots, so
`ENST00000155840.KCNQ1.cleanLb_hmm_manual.fasta` is KCNQ1; the leading code is
the human reference transcript the alignment is anchored on.

Genes included: the four HCN pacemaker channels (HCN1 to HCN4), the potassium
channels KCNQ1, KCNE1, KCNH2 and KCNE2, the calcium channels CACNA1C, CACNA1D,
CACNA1G, CACNA1H, CACNA1I with their auxiliary subunits CACNA2D1, CACNB1,
CACNB2, CACNB3 and CACNG1, and the sodium-calcium exchanger SLC8A1.

Not every species is present in every gene; a species missing from a gene's file
simply has no calls for that gene's variants.

## `species_info.csv`

One row per species, 153 rows.

| Column | Meaning |
|---|---|
| `genome_id` | Genome assembly identifier; the key that ties this file to the alignments and the trees |
| `scientific_name` | Latin name, used in all plots and tables |
| `common_name` | Everyday name |
| `heart_rate_bpm` | Resting heart rate in beats per minute, compiled from the published literature |
| `body_mass_g` | Adult body mass in grams, from the PHYLACINE 1.2 database |
| `order`, `family`, `genus` | Taxonomy, also from PHYLACINE 1.2 |

One species, *Nannospalax galili*, has no body mass and is left out of runs that
treat body mass as a confounder.

## `trees/`

| File | What it is |
|---|---|
| `ebisuya_tree.nwk` | The tree the program uses. Supplied for this project and covering all 153 species. Its file gives lengths for the internal branches only, so each final branch is run out to the present before the covariance matrix is built; the run notes say so. It gives a divergence time of zero for one pair, *Microtus arvalis* and *Microtus agrestis*, so the tree cannot tell those two apart and the comparison between them carries no weight. |

Any other tree can be used instead: press **Use my own tree file...** on the Data
settings tab. Newick files (`.nwk`, `.tre`, `.tree`, `.newick`) and covariance
matrices saved as `.csv`, with species names as both the first row and the first
column, are both accepted. A covariance matrix gives identical results to the
equivalent Newick file but cannot be cut into clades, because the branching is no
longer in it.

### `trees/other_trees/`

Three further trees, kept for comparison. None is used unless you browse to it.

| File | What it is |
|---|---|
| `upham_dated_tree.nwk` | Pruned from one of the 10,000 posterior trees of Upham et al. (PLOS Biology), taken from the megatrees archive. Time-calibrated in millions of years: 152 distinct branching dates, real terminal branches, all 153 species. Independent of the genes being tested. |
| `upham_dated_tree_species_matching.csv` | Which tip of that tree each of the 153 species was matched to, and how. 134 matched exactly, 13 are subspecies matched to their species, 5 are naming changes, and 1 (*Nannospalax galili*) is approximate: that genus is absent, so it sits at its species complex, *Spalax ehrenbergi*. |
| `upham_mammal_tree.nwk` | The same phylogeny as supplied with the earlier data, rebuilt from the covariance matrix in `external_tree.npz`. Its branching is resolved but its dating is not: all 151 branching events sit on just six dates. 152 species. |
| `sequence_upgma_tree.nwk` | Built from the gene sequences themselves. Not independent of the data being tested, so results from it should be read with that in mind. |

## Where the numbers originally come from

1. **Gene alignments** across mammalian genomes, anchored on human reference
   transcripts. These give the amino acid each species carries at each position.
2. **Published resting heart rates**, compiled per species from the literature.
3. **PHYLACINE 1.2 / Upham et al.**, for the phylogeny, the taxonomy and body mass.
