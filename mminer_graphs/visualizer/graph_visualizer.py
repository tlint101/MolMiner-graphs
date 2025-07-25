"""
Functions to visualize molecular graphs.
Adapted from MMGX (https://github.com/ohuelab/MMGX.git)
Paper: "Enhancing property and activity prediction and interpretation using multiple molecular graph representations with MMGX"
Communications Chemistry (2024), https://doi.org/10.1038/s42004-024-01084-4
"""

import random
import io
from PIL import Image
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from mminer_graphs.graph import MolGraph, AtomGraph
from typing import Union

__all__ = ["Graph"]


class Graph(MolGraph):
    def __init__(self, smi):
        super().__init__(smi)
        self.smi = smi

    # label atoms with indices for visual
    def get_mol(self, mol: Chem.rdchem.Mol = None, atom_index=True, savepath=None):
        """
        Plot an RDKit Molecule with atom index.
        :param mol: Chem.rdchem.Mol
            Input RDKit molecule object.
        :param atom_index: bool
            Set atom index labels on molecule.
        :param savepath: str
            Set save path of image.
        :return:
        """
        # check ROMol
        if mol is None:
            mol = Chem.MolFromSmiles(self.smi)
        # get atom index
        if atom_index == True:
            for atom in mol.GetAtoms():
                atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

        if savepath:
            Draw.MolToFile(mol, savepath)
        return mol

    # Final display function: one matplotlib figure, side-by-side
    def display_graph(self, moleculegraph: MolGraph, colors: Union[str, list] = None, graph_labels: bool = True,
                      titles: list = None, titlesize: int = 15, labelsize: int = 10, figsize: tuple = (10, 4),
                      savefig: str = None):
        """
        Draw a query molecule and its associated graph. The graph is made using AtomGraph or PharmacophoreGraph.
        :param moleculegraph: MolGraph
            Graph information genreated from AtomGraph or PharmacophoreGraph.
        :param colors: Union[str,list]
            Input color conditions for the molecule graph.
        :param graph_labels: bool
            Set the atom labels for the graph.
        :param titles: list
            Input title names for the molecule and graph type.
        :param titlesize: int
            Set the size of the graph titles
        :param labelsize: int
            Set the size of the graph labels
        :param figsize: tuple
            Set size of the figure
        :param savefig: str
            File path to save the image.
        :return:
        """
        # extract attributes to moleculegraph
        mol = moleculegraph.mol
        cliques = moleculegraph.cliques
        edges = moleculegraph.edges

        # get color palette
        colors = _get_color_palette(cliques, colors)

        # Get layout from RDKit 2D coordinates
        pos = _get_clique_positions(mol, cliques)

        # generate list for bond highlights from the graph cliques
        if not isinstance(moleculegraph, AtomGraph):
            bond_highlights = _get_rdkit_bond_highlights_from_cliques(mol, cliques)
        else:
            bond_highlights = None

        # generate RDKit image
        mol_img = self._get_rdkit_image(mol, cliques, colors, label=graph_labels, highlight_bonds=bond_highlights,
                                        size=(1000, 1000))

        # create molecular graph using NetworkX
        G = nx.Graph()
        for i in range(len(cliques)):
            G.add_node(i)
        for u, v in edges:
            G.add_edge(u, v)
        node_colors = [colors[i] for i in G.nodes]

        # place figures side by side
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # get plot titles
        if titles is None:
            graph_type = _display_title_for_object(moleculegraph)
            titles = ['Molecule', graph_type]

        # left - RDKit image
        axs[0].imshow(mol_img)
        axs[0].axis("off")
        axs[0].set_title(titles[0], fontsize=titlesize)

        # right - NetworkX graph
        nx.draw(
            G, pos=pos, ax=axs[1],
            with_labels=graph_labels,
            node_color=node_colors,
            node_size=500,
            font_size=labelsize
        )
        axs[1].axis("off")
        axs[1].set_aspect('equal', adjustable='datalim')  # graph scaling
        axs[1].autoscale_view()  # graph scaling
        axs[1].set_title(titles[1], fontsize=titlesize)

        plt.tight_layout()
        plt.show()

        # save figure
        if savefig:
            fig.savefig(savefig, dpi=300)

        plt.close(fig)

    def _get_rdkit_image(self, mol, cliques, colors, label, highlight_bonds, size=(500, 500)):
        """Get an image from RDKit with atoms highlighted accordingly"""
        mol = self.get_mol(mol)
        AllChem.Compute2DCoords(mol)

        atom_colors = {}
        highlight_atoms = []
        for i, clique in enumerate(cliques):
            for idx in clique:
                atom_colors[idx] = tuple(colors[i][:3])
                highlight_atoms.append(idx)

        d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        options = d2d.drawOptions()

        if not label:
            # Remove atom labels by overriding them with empty strings
            for atom in mol.GetAtoms():
                options.atomLabels[atom.GetIdx()] = ""

        rdMolDraw2D.PrepareAndDrawMolecule(
            d2d,
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=highlight_bonds
        )
        d2d.FinishDrawing()
        png = d2d.GetDrawingText()
        img = Image.open(io.BytesIO(png))
        return img


def _display_title_for_object(obj):
    """support method to obtain object name for plotting title"""
    class_name = obj.__class__.__name__
    title = f"{class_name}"
    if title == "JunctionTreeGraph":
        title = title.replace("TreeGraph", " Tree Graph")
    else:
        title = title.replace("Graph", " Graph")

    return title


def _get_color_palette(cliques, colors):
    """support method to get color palette for graph visualization"""
    # random
    if colors is None:
        colors = [[random.random(), random.random(), random.random()] for _ in cliques]

    # seaborn color palette
    elif isinstance(colors, str):
        try:
            colors = sns.color_palette(colors, n_colors=len(cliques))
        except ValueError as e:
            raise ValueError(f"'{colors}' is not a valid seaborn color palette!") from e

    # a list of colors
    elif isinstance(colors, list):
        try:
            # Convert color strings/hex to RGBA
            colors = [mcolors.to_rgba(c) for c in colors]
        except ValueError as e:
            raise ValueError("Invalid color in list: ensure all are valid color names or hex codes.") from e
    else:
        raise ValueError("`colors` must be None, a seaborn palette name, or a list of color strings.")
    return colors


def _get_clique_positions(mol, cliques):
    """support method to obtain clique positions from 2D atom coordinates"""
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = {}
    for i, clique in enumerate(cliques):
        xs, ys = [], []
        for idx in clique:
            x, y = conf.GetAtomPosition(idx).x, conf.GetAtomPosition(idx).y
            xs.append(x)
            ys.append(y)
        pos[i] = (sum(xs) / len(xs), sum(ys) / len(ys))
    return pos


def _get_rdkit_bond_highlights_from_cliques(mol, cliques):
    """support method to extract bond highlights based on the cliques"""
    highlight_bonds = set()

    for clique in cliques:
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                a1, a2 = clique[i], clique[j]
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is not None:
                    highlight_bonds.add(bond.GetIdx())

    return list(highlight_bonds)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
