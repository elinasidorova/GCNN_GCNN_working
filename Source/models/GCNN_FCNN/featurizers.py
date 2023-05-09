import copy
import json
import random
import warnings

import dgl
import torch
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from torch_geometric.utils import from_networkx

from config import ROOT_DIR


class DGLFeaturizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def featurize(self, mol, require_node_features=True, require_edge_features=True):
        dgl_graph = mol_to_bigraph(mol, **self.kwargs)
        networkx_graph = dgl.to_networkx(dgl_graph)
        graph = from_networkx(networkx_graph)
        if 'h' not in dgl_graph.ndata:
            if require_node_features:
                warnings.warn(f"can't featurize {Chem.MolToSmiles(mol)}: 'h' not in graph.ndata. Skipping.")
                return None
            else:
                warnings.warn(f"No node_features in {Chem.MolToSmiles(mol)}")
                dgl_graph.ndata['h'] = None
        if 'e' not in dgl_graph.edata:
            if require_edge_features:
                warnings.warn(f"can't featurize {Chem.MolToSmiles(mol)}: 'e' not in graph.edata. Skipping.")
                return None
            else:
                warnings.warn(f"No edge_features in {Chem.MolToSmiles(mol)}")
                dgl_graph.edata['e'] = None
            return None
        graph.x = dgl_graph.ndata['h']
        graph.edge_attr = dgl_graph.edata['e']
        graph.id = None
        return graph


class SkipatomFeaturizer:
    """
    Class for extracting element features by skipatom_models approach

    Attributes
    ----------
    get_vector : dict
        skipatom vectors for each element

    Methods
    ----------
    _featurize(element : str)
        get skipatom features for given element
    """

    def __init__(self, vectors_filename=ROOT_DIR / "Source/models/GCNN_FCNN/skipatom_vectors_dim200.json"):
        with open(vectors_filename, "r") as f:
            self.get_vector = json.load(f)

    def featurize(self, element):
        """

        Parameters
        ----------
        element : str
            element to be featurized

        Returns
        -------
            features : torch.tensor
                features of an element obtained from skipatom approach, shape (1, 200)
        """
        return torch.tensor(self.get_vector[element]).unsqueeze(0)


def featurize_sdf_with_metal(path_to_sdf, mol_featurizer, metal_featurizer, seed=42):
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
    metal_featurizer : featurizer, optional
        instance of the class used for extracting metal features

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    mols = [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    mol_graphs = [mol_featurizer.featurize(m) for m in mols]

    all_data = []
    for mol, graph in zip(mols, mol_graphs):
        targets = [prop for prop in mol.GetPropNames() if prop.startswith("logK_")]
        for target in targets:
            new_graph = copy.deepcopy(graph)

            element_symbol = target.split("_")[-1]
            new_graph.metal_x = metal_featurizer.featurize(element_symbol)
            new_graph.y = torch.tensor([[float(mol.GetProp(target))]])
            all_data += [new_graph]
    random.Random(seed).shuffle(all_data)

    return all_data


def featurize_sdf_with_metal_and_conditions(path_to_sdf, mol_featurizer, metal_featurizer, seed=42):
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
    metal_featurizer : featurizer, optional
        instance of the class used for extracting metal features
    data_multy_coefficients: dict
        each complex of metal Me will be used data_multy_coefficients[Me] times

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    mols = [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    mol_features = [mol_featurizer.featurize(m) for m in mols]

    all_data = []
    for mol_ind in range(len(mols)):
        metals = []
        conditions = []
        logKs = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("logK_")]:
            element_symbol, charge_str, temperature_str, ionic_str_str = target.split("_")[1:]
            charge = float(charge_str.split("=")[-1])
            temperature = float(temperature_str.split("=")[-1])
            ionic_str = float(ionic_str_str.split("=")[-1])
            metals += [element_symbol]
            conditions += [torch.tensor([[charge, temperature, ionic_str]])]
            logKs += [float(mols[mol_ind].GetProp(target))]
        for element_symbol, condition_values, logK in zip(metals, conditions, logKs):
            features = copy.deepcopy(mol_features[mol_ind])
            features.metal_x = torch.cat((metal_featurizer.featurize(element_symbol), condition_values), dim=-1)
            features.y = torch.tensor([[logK]])
            all_data += [features]
    random.Random(seed).shuffle(all_data)

    return all_data
