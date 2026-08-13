"""
hrv_variants: finding genetic variants linked to heart rate across mammals.

The modules, in the order the pipeline uses them:

  config              every option the user can set, in one Settings object
  sequences           reads DNA alignments and calls amino-acid variants
  trees               reads phylogenetic trees and builds covariance matrices
  clades              the four ways of grouping species into clades
  data_io             loads everything, caches it, handles uncalled genotypes
  stats_tools         small shared statistical helpers
  method_pgls         Method 1: phylogenetic generalized least squares
  method_clade_sharing Method 2: the clade-sharing method
  method_xgboost      Method 3: the machine learning model (XGBoost) and its search
  method_combined     the rule that keeps only variants both tests find
  explore_dnds        dN/dS per gene
  explore_pca         the protein-space map
  plots               every figure
  reporting           the output tables
  pipeline            ties it together; this is what the window calls
  gui                 the window itself
"""

__version__ = "1.0"
