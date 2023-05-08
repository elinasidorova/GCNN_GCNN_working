import copy
import random

import torch
from rdkit import Chem


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
        if graph is None: continue
        targets = [prop for prop in mol.GetPropNames() if prop.startswith("logK_")]
        for target in targets:
            new_graph = copy.deepcopy(graph)

            element_symbol = target.split("_")[-1]
            new_graph.u = metal_featurizer.featurize(element_symbol)
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
            new_graph.u = torch.cat((metal_featurizer.featurize(metal),
                                     torch.tensor([[temperature, ionic_str, charge]])), dim=-1)
            new_graph.y = torch.tensor([[logK]])
            all_data += [new_graph]
    random.Random(seed).shuffle(all_data)

    return all_data
