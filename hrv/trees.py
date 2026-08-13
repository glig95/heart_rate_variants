"""
Reading phylogenetic trees and turning them into a covariance matrix.

Why a covariance matrix: closely related species are not independent data
points, so a plain correlation between a variant and heart rate can be driven
purely by shared ancestry. The covariance matrix says, for every pair of
species, how much evolutionary history they share. PGLS uses it to discount
that shared history.

The number for a pair of species is the length of the path from the root of
the tree down to their most recent common ancestor. Two species that split
recently share almost the whole path and get a high number; two species from
different branches of the tree share only the root and get a low number.
"""

from pathlib import Path
import numpy as np
import pandas as pd


class Node:
    """One point in a tree: either a species (a tip) or a common ancestor."""

    def __init__(self, name=None, length=0.0):
        self.name = name
        self.length = length      # branch length leading to this node
        self.children = []


def _strip_comments_and_space(text):
    """
    Remove square-bracket comments and layout whitespace from Newick text.

    Tree files exported by other programs often carry annotations in square
    brackets, as in "A:1[&&NHX:support=100]", and are wrapped across lines for
    readability. Neither is part of the tree, but both will derail a parser that
    reads the file character by character. Whitespace inside a quoted species
    name is kept, because there it is part of the name.
    """
    out = []
    depth = 0
    quote = None
    for character in text:
        if quote:
            out.append(character)
            if character == quote:
                quote = None
            continue
        if character in "'\"":
            quote = character
            out.append(character)
            continue
        if character == "[":
            depth += 1
            continue
        if character == "]":
            depth = max(0, depth - 1)
            continue
        if depth == 0 and not character.isspace():
            out.append(character)
    return "".join(out)


def read_newick(path):
    """
    Read a tree written in Newick format, the standard text format for trees.

    A Newick file looks like "((mouse:1,rat:1):3,human:4);" where the numbers
    are branch lengths. Returns the root node of the tree.

    A branch length that cannot be read as a number stops the run with an error
    rather than being taken as zero. Silently reading a length as zero would
    leave a tree with no shape at all, and the phylogenetic correction would
    then quietly do nothing while still appearing to work.
    """
    text = _strip_comments_and_space(Path(path).read_text())
    text = text.split(";")[0]          # everything before the first ";" is the tree
    if not text:
        raise ValueError(f"{Path(path).name} contains no tree.")
    position = [0]

    def parse_node():
        node = Node()
        if text[position[0]] == "(":
            position[0] += 1           # step past "("
            while True:
                node.children.append(parse_node())
                if position[0] < len(text) and text[position[0]] == ",":
                    position[0] += 1
                else:
                    break
            if position[0] >= len(text) or text[position[0]] != ")":
                raise ValueError(f"{Path(path).name} is not a valid Newick tree: "
                                 f"a closing bracket is missing.")
            position[0] += 1           # step past ")"
        # read the label (a species name for tips, support values for ancestors)
        if position[0] < len(text) and text[position[0]] in "'\"":
            quote = text[position[0]]
            position[0] += 1
            start = position[0]
            while position[0] < len(text) and text[position[0]] != quote:
                position[0] += 1
            label = text[start:position[0]]
            position[0] += 1           # step past the closing quote
        else:
            start = position[0]
            while position[0] < len(text) and text[position[0]] not in "(),:":
                position[0] += 1
            label = text[start:position[0]]
        if not node.children:
            node.name = label
        # read the branch length after ":"
        if position[0] < len(text) and text[position[0]] == ":":
            position[0] += 1
            start = position[0]
            while position[0] < len(text) and text[position[0]] not in "(),":
                position[0] += 1
            written = text[start:position[0]]
            try:
                node.length = float(written)
            except ValueError:
                raise ValueError(
                    f"{Path(path).name} has a branch length that is not a number: "
                    f"\"{written}\". Every \":\" in a Newick file must be followed by "
                    f"a number.") from None
        return node

    root = parse_node()
    if position[0] < len(text):
        raise ValueError(f"{Path(path).name} is not a valid Newick tree: there is "
                         f"text after the end of the tree.")
    return root


def list_tips(node):
    """Return every species name below this node, in the order the tree lists them."""
    if not node.children:
        return [node.name]
    names = []
    for child in node.children:
        names.extend(list_tips(child))
    return names


def tip_depths(root):
    """Return the distance from the root to every species in the tree."""
    depths = {}

    def walk(node, so_far):
        depth = so_far + node.length
        if not node.children:
            depths[node.name] = depth
        for child in node.children:
            walk(child, depth)

    walk(root, 0.0)
    return depths


def terminal_branches(root):
    """Return the length of every branch that ends in a species."""
    lengths = []

    def walk(node):
        if not node.children:
            lengths.append(node.length)
        for child in node.children:
            walk(child)

    walk(root)
    return lengths


def extend_tips_to_present(root):
    """
    Stretch every species' final branch so that all of them end at the same time.

    Some tree files give branch lengths only for the internal branches, leaving
    every species with a final branch of length zero. Taken literally that says
    two sister species sit at exactly the same point, which makes the covariance
    matrix impossible to invert. Since the species in this kind of dataset are
    all alive today, the repair is to run each final branch out to the present.

    This is only ever applied to a tree whose final branches are all missing;
    see `load_tree_covariance`. It is not applied to a tree that carries real
    tip lengths, because stretching those would change how strongly the method
    treats each pair of species as related, which is the very thing the tree is
    there to supply. PGLS itself does not require every species to end at the
    same point.

    Returns how many branches were stretched.
    """
    depths = tip_depths(root)
    if not depths:
        return 0
    present = max(depths.values())

    changed = 0

    def walk(node, so_far):
        nonlocal changed
        depth = so_far + node.length
        if not node.children:
            shortfall = present - depth
            if shortfall > 1e-12:
                node.length += shortfall
                changed += 1
            return
        for child in node.children:
            walk(child, depth)

    walk(root, 0.0)
    return changed


def covariance_from_tree(root):
    """
    Build the shared-ancestry covariance matrix from a tree.

    Walks down from the root keeping track of the distance traveled. Each
    ancestor writes its own depth into every pair of species that meet there,
    meaning pairs drawn from two different branches below it, so each pair ends
    up holding the depth of their most recent common ancestor. The diagonal
    holds each species' full distance from the root.

    Pairs are taken branch against branch rather than across everything below,
    so a pair that already met deeper in the tree is never overwritten. Doing it
    the other way would need depth to increase all the way down, which fails on
    trees that carry negative branch lengths, as neighbor-joining trees often
    do.

    Returns the matrix as a pandas DataFrame, with species names on both axes.
    """
    names = list_tips(root)
    index = {n: i for i, n in enumerate(names)}
    size = len(names)
    matrix = np.zeros((size, size))

    def walk(node, depth_so_far):
        depth = depth_so_far + node.length
        if not node.children:
            matrix[index[node.name], index[node.name]] = depth
            return [index[node.name]]
        branches = [walk(child, depth) for child in node.children]
        for first in range(len(branches)):
            for second in range(first + 1, len(branches)):
                for i in branches[first]:
                    for j in branches[second]:
                        matrix[i, j] = matrix[j, i] = depth
        return [tip for branch in branches for tip in branch]

    walk(root, 0.0)
    return pd.DataFrame(matrix, index=names, columns=names)


def load_tree_covariance(path, notes=None):
    """
    Load a tree from disk and return its covariance matrix.

    Two file types are accepted:
      * a Newick tree file (.nwk, .tre, .tree, .newick, .txt)
      * a covariance matrix saved directly as a .csv, with species names as
        both the first row and the first column

    A tree whose final branches are all missing has them run out to the present
    first, and a line saying so is added to `notes` if one was passed in. A tree
    that carries real tip lengths is used exactly as it stands.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        matrix = pd.read_csv(path, index_col=0)
        matrix.columns = [str(c) for c in matrix.columns]
        matrix.index = [str(i) for i in matrix.index]
        _check_usable(matrix, path)
        return matrix

    root = read_newick(path)
    tips = terminal_branches(root)
    if tips and max(tips) <= 0:
        stretched = extend_tips_to_present(root)
        if stretched and notes is not None:
            notes.append(
                f"This tree gives no length for any of its {stretched} final branches, "
                f"so each one was run out to the present, which is where living species "
                f"belong. Without this every pair of sister species would sit at exactly "
                f"the same point and the phylogenetic correction could not be computed. "
                f"Internal branches are untouched.")
    matrix = covariance_from_tree(root)
    _check_usable(matrix, path)
    _warn_about_identical_species(matrix, notes)
    return matrix


def identical_species(matrix, tolerance=1e-9):
    """
    Find pairs of species the tree places at exactly the same point.

    This happens when a tree gives a divergence time of zero, so the two species
    sit on top of one another and nothing in the tree distinguishes them. Their
    rows in the covariance matrix are then identical.
    """
    values = np.asarray(matrix, dtype=float)
    diagonal = np.sqrt(np.diag(values))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = values / np.outer(diagonal, diagonal)
    names = list(matrix.index) if hasattr(matrix, "index") else list(range(len(values)))
    pairs = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if np.isfinite(correlation[i, j]) and correlation[i, j] > 1 - tolerance:
                pairs.append((names[i], names[j]))
    return pairs


def _warn_about_identical_species(matrix, notes):
    """Record any species the tree cannot tell apart, so the run says so plainly."""
    if notes is None:
        return
    pairs = identical_species(matrix)
    if not pairs:
        return
    listed = "; ".join(f"{a} and {b}" for a, b in pairs[:5])
    if len(pairs) > 5:
        listed += f"; and {len(pairs) - 5} more pairs"
    notes.append(
        f"This tree gives a divergence time of zero for {len(pairs)} pair(s) of species "
        f"({listed}), so it places them at exactly the same point and says nothing about "
        f"how they differ. The comparison between each such pair carries no weight in the "
        f"phylogenetic test, which loses one independent observation per pair. Every other "
        f"comparison is unaffected.")


def _check_usable(matrix, path):
    """
    Stop the run if a tree carries no usable branch lengths at all.

    An all-zero matrix says every species sits at the root, which is not a
    phylogeny. Left alone it would sail through as a matrix of near-zeros and
    the phylogenetic correction would quietly become no correction at all, so
    it is better to say so plainly.
    """
    diagonal = np.diag(np.asarray(matrix, dtype=float))
    if len(diagonal) == 0 or float(np.nanmax(np.abs(diagonal))) <= 0:
        raise ValueError(
            f"{Path(path).name} gives no branch lengths, so it says nothing about how "
            f"closely the species are related and cannot be used for a phylogenetic "
            f"correction.")


def cut_tree_into_clades(root, n_clades):
    """
    Split a tree into a chosen number of groups.

    Starts with the whole tree as one group, then repeatedly opens up whichever
    group holds the most species, until the requested number of groups is
    reached. Every group is one whole branch, so the groups are monophyletic.

    A branch that splits three or more ways is opened all at once, so the final
    count can come out slightly above the request. The clades table written with
    every run always shows the groups that were actually used.

    Returns a dictionary of {species name: clade name}.
    """
    groups = [root]
    while len(groups) < n_clades:
        splittable = [g for g in groups if g.children]
        if not splittable:
            break
        biggest = max(splittable, key=lambda g: len(list_tips(g)))
        groups.remove(biggest)
        groups.extend(biggest.children)

    labels = {}
    for number, group in enumerate(groups, start=1):
        for tip in list_tips(group):
            labels[tip] = f"clade_{number:02d}"
    return labels


