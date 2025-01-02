import json
import random
import warnings

from dgllife.utils import BaseAtomFeaturizer, ConcatFeaturizer, atom_type_one_hot, atom_degree_one_hot, \
    atom_implicit_valence_one_hot, atom_formal_charge, atom_num_radical_electrons, atom_hybridization_one_hot, \
    atom_is_aromatic, atom_total_num_H_one_hot
from rdkit import Chem
from torch_geometric.data import Data

from config import ROOT_DIR
import torch


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_mol':
            return self.x_mol.size(0)
        if key == 'edge_index_metal':
            return self.x_metal.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def get_skipatom_features(atom, vectors_filename=ROOT_DIR / "Source/models/GCNN_FCNN/skipatom_vectors_dim200.json"):
    with open(vectors_filename, "r") as f:
        get_vector = json.load(f)
    return get_vector[atom.GetSymbol()]


class DglMetalFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_field='h'):
        super(DglMetalFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [get_skipatom_features,
                 atom_type_one_hot,
                 atom_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_total_num_H_one_hot]
            )})


def featurize_df(df, mol_featurizer, metal_featurizer, seed=42):
    all_data = []
    for i in df.index:
        mol = Chem.MolFromSmiles(df["smiles"][i])
        metal = Chem.MolFromSmiles(df["metal"][i])
        if mol is None:
            warnings.warn(f"Can't read mol from smiles: {df['smiles']}")
            continue
        if metal is None:
            warnings.warn(f"Can't read mol from smiles: {df['metal']}")
            continue

        mol_graph = mol_featurizer.featurize(mol)
        metal_graph = metal_featurizer.featurize(metal)
        if mol_graph is None:
            warnings.warn(f"Can't featurize: {df['smiles']}")
            continue
        if metal_graph is None:
            warnings.warn(f"Can't featurize: {df['metal']}")
            continue

        all_data += [PairData(x_mol=mol_graph.x, edge_index_mol=mol_graph.edge_index,
                              edge_attr_mol=mol_graph.edge_attr, u_mol=mol_graph.u,
                              x_metal=metal_graph.x, edge_index_metal=metal_graph.edge_index,
                              edge_attr_metal=metal_graph.edge_attr, u_metal=metal_graph.u,
                              y=df["logK"][i])]
    random.Random(seed).shuffle(all_data)

    return all_data


class PairDataSolubility(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_solvent': #edge_index_mol -> edge_index_solvent
            return self.x_solvent.size(0) #self.x_mol.size -> self.x_solvent.size
        if key == 'edge_index_molecule': #edge_index_metal -> edge_index_molecule
            return self.x_molecule.size(0) #self.x_metal.size -> self.x_molecule.size
        return super().__inc__(key, value, *args, **kwargs)


def featurize_sdf_mol_solv(path_to_sdf, solvent_featurizer, molecule_featurizer, seed=42):


    if path_to_sdf is None:
        raise ValueError("'path_to_sdf' parameter should be stated")

    mols = [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]

    
    all_data = []
    for mol_ind in range(len(mols)):

        molecule = mols[mol_ind]
        solvent = Chem.MolFromSmiles(molecule.GetProp('Solvent_smiles'))

        solvent_graph = solvent_featurizer.featurize(solvent)
        molecule_graph = molecule_featurizer.featurize(molecule)

        if solvent_graph is None:
            warnings.warn(f"Can't featurize solvent: {mol_ind}")
            continue
        if molecule_graph is None:
            warnings.warn(f"Can't featurize molecule: {mol_ind}")
            continue


        all_data.append(PairDataSolubility(x_solvent=solvent_graph.x, edge_index_solvent=solvent_graph.edge_index,
                              x_molecule=molecule_graph.x, edge_index_molecule=molecule_graph.edge_index,
                              y=torch.Tensor([float(molecule.GetProp('Solubility'))]), batch_solvent=torch.tensor([0] * solvent_graph.x.size(0)),
                batch_molecule=torch.tensor([1] * molecule_graph.x.size(0))))


    random.Random(seed).shuffle(all_data)

    return all_data
