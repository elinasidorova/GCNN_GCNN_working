import random
import warnings

import dgl
import torch
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from torch_geometric.utils import from_networkx

from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer


class DGLFeaturizer:
    def __init__(self, require_node_features=True, require_edge_features=True, **kwargs):
        self.require_node_features = require_node_features
        self.require_edge_features = require_edge_features
        self.kwargs = kwargs

    def featurize(self, mol):
        dgl_graph = mol_to_bigraph(mol, **self.kwargs)
        networkx_graph = dgl.to_networkx(dgl_graph)
        graph = from_networkx(networkx_graph)
        if 'h' not in dgl_graph.ndata:
            if self.require_node_features:
                warnings.warn(f"can't featurize {Chem.MolToSmiles(mol)}: 'h' not in graph.ndata. Skipping.")
                return None
            else:
                dgl_graph.ndata['h'] = torch.zeros((dgl_graph.num_nodes(), 1))
        if 'e' not in dgl_graph.edata:
            if self.require_edge_features:
                warnings.warn(f"can't featurize {Chem.MolToSmiles(mol)}: 'e' not in graph.edata. Skipping.")
                return None
            else:
                dgl_graph.edata['e'] = torch.zeros((dgl_graph.num_edges(), 1))
        graph.x = dgl_graph.ndata['h']
        graph.edge_attr = dgl_graph.edata['e']
        graph.id = None
        return graph


def featurize_sdf(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(), seed=42):
    """
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    if (path_to_sdf is None) and (molecules is None):
        raise ValueError("'path_to_sdf' or 'molecules' parameter should be stated, got neither")
    elif (path_to_sdf is not None) and (molecules is not None):
        raise ValueError("Only one source ('path_to_sdf' or 'molecules' parameter) should be stated, got both")
    mols = molecules or [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    mol_graphs = [mol_featurizer.featurize(m) for m in mols]

    all_data = []
    for mol, graph in zip(mols, mol_graphs):
        targets = [prop for prop in mol.GetPropNames() if prop.startswith("logK")]
        if len(targets) > 1: raise ValueError("Several targets for molecule")
        graph.y = {"logK": torch.tensor([[float(mol.GetProp(targets[0]))]])}
        all_data += [graph]
    random.Random(seed).shuffle(all_data)

    return all_data
