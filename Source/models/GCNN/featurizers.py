import dgl
import numpy as np
import pandas as pd
import torch
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx


class VecData(Dataset):
    def __init__(self, x=None, y=None):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def one_of_k_encoding(x, allowable_set):
    """Encodes elements of a provided set as integers.

      Parameters
      ----------
      x: object
        Must be present in `allowable_set`.
      allowable_set: list
        List of allowable quantities.

      Returns
      -------
      encoding: list
        encoding[i] == 1 if allowable_set[i] == x
        encoding[i] == 0 otherwise
        For example, [True, False, False].

      Raises
      ------
      `ValueError` if `x` is not in `allowable_set`.
    """
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element.

      Unlike `one_of_k_encoding`, if `x` is not in `allowable_set`, this method
      pretends that `x` is the last element of `allowable_set`.

      Parameters
      ----------
      x: object
        Must be present in `allowable_set`.
      allowable_set: list
        List of allowable quantities.

      Returns
      --------
      encoding: list
        encoding[i] == 1 if allowable_set[i] == x or x not in allowable_set
        encoding[i] == 0 otherwise
        For example, [True, False, False]. For example, [False, False, True].
      """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def adj_to_edge(adj_list: list) -> torch.Tensor:
    """
    Convert adjacency list of a graph to list of its edges

    Parameters
    ----------
    adj_list : list
        Adjacency list of a graph, i.e. adj_list[i] is a list of neighbors of i-th vertex

    Returns
    -------
    edge_list : list
       List of graph edges in format [start_vertex, end_vertex]
    """
    edge_list = []
    for i, current_neighbours in enumerate(adj_list):
        for neighbour in current_neighbours:
            edge_list.append([i, neighbour])

    return torch.tensor(edge_list, dtype=torch.int64)


def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
    """
    Helper method used to compute per-atom feature vectors.
    Many different featurization methods compute per-atom features such as ConvMolFeaturizer, WeaveFeaturizer. This method computes such features.

    Parameters
    ----------
    bool_id_feat: bool, optional
    Return an array of unique identifiers corresponding to atom type.
    explicit_H: bool, optional
    If true, model hydrogens explicitly
    use_chirality: bool, optional
    If true, use chirality information.

    Returns
    -------
    np.ndarray of per-atom features.
    """
    if bool_id_feat:
        return  # np.array([atom_to_id(atom)])
    else:
        from rdkit import Chem
        results = one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                'C',
                'N',
                'O',
                'S',
                'F',
                'Si',
                'P',
                'Cl',
                'Br',
                'Mg',
                'Na',
                'Ca',
                'Fe',
                'As',
                'Al',
                'I',
                'B',
                'V',
                'K',
                'Tl',
                'Yb',
                'Sb',
                'Sn',
                'Ag',
                'Pd',
                'Co',
                'Se',
                'Ti',
                'Zn',
                'H',  # H? H!
                'Li',
                'Ge',
                'Cu',
                'Au',
                'Ni',
                'Cd',
                'In',
                'Mn',
                'Zr',
                'Cr',
                'Pt',
                'Hg',
                'Pb',
                'Unknown'
            ]) + one_of_k_encoding(atom.GetDegree(),
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)


class DGLFeaturizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def featurize(self, mol):
        dgl_graph = mol_to_bigraph(mol, **self.kwargs)
        networkx_graph = dgl.to_networkx(dgl_graph)
        graph = from_networkx(networkx_graph)
        graph.x = dgl_graph.ndata['h'] if 'h' in dgl_graph.ndata else None
        graph.edge_attr = dgl_graph.edata['e'] if 'e' in dgl_graph.edata else None
        graph.id = None
        return graph


def featurize_df(df: pd.DataFrame, mol_featurizer, target="logK"):
    molecules = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s))
    graphs = molecules.apply(lambda m: mol_featurizer.featurize(m)).tolist()
    targets = df[target].tolist()
    for graph, target_value in zip(graphs, targets):
        graph.y = {target: torch.tensor([[float(target_value)]])}

    return graphs
