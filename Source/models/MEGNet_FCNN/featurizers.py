import copy
import json
import random

import dgl
import torch
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from torch_geometric.utils import from_networkx


class DGLFeaturizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def featurize(self, mol):
        dgl_graph = mol_to_bigraph(mol, **self.kwargs)
        networkx_graph = dgl.to_networkx(dgl_graph)
        graph = from_networkx(networkx_graph)
        if 'h' not in dgl_graph.ndata or 'e' not in dgl_graph.edata:
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

    def __init__(self, vectors_filename="skipatom_vectors_dim200.json"):
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


def featurize_sdf_with_metal_and_conditions(path_to_sdf, mol_featurizer, metal_featurizer, z_in_metal=False, seed=42):
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
    mol_graphs = [mol_featurizer.featurize(m) for m in mols]

    all_data = []
    for mol, graph in zip(mols, mol_graphs):
        if graph is None: continue
        metals = []
        charges = []
        temperatures = []
        ionic_strs = []
        logKs = []
        for target in [prop for prop in mol.GetPropNames() if prop.startswith("logK_")]:
            element_symbol, charge_str, temperature_str, ionic_str_str = target.split("_")[1:]
            charges += [float(charge_str.split("=")[-1])]
            temperatures += [float(temperature_str.split("=")[-1])]
            ionic_strs += [float(ionic_str_str.split("=")[-1])]
            metals += [element_symbol]
            logKs += [float(mol.GetProp(target))]
        for metal, charge, temperature, ionic_str, logK in zip(metals, charges, temperatures, ionic_strs, logKs):
            new_graph = copy.deepcopy(graph)
            if z_in_metal:
                new_graph.u = torch.tensor([[temperature, ionic_str]])
                new_graph.metal_x = torch.cat((metal_featurizer.featurize(metal), torch.tensor([[charge]])), dim=-1)
            else:
                new_graph.u = torch.tensor([[temperature, ionic_str, charge]])
                new_graph.metal_x = metal_featurizer.featurize(metal)
            new_graph.y = torch.tensor([[logK]])
            all_data += [new_graph]
    random.Random(seed).shuffle(all_data)

    return all_data
