import copy
import random

import numpy as np
import torch
from loguru import logger
from rdkit import Chem
from rdkit.Chem import (
    AllChem, DataStructs
)
from torch.utils.data import Dataset
from torch_geometric.data import Data


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


def drop_edge_duplicates(edge_list: list):
    """
    Drop edge duplicates.

    Parameters
    ----------
    edge_list : list
        List of graph edges.

    Returns
    -------
    edge_list : list
        List of graph edges without duplicates.
    """
    ids_to_drop = []
    for start, current_pair in enumerate(edge_list):
        for i, target_pair in enumerate(edge_list[start:]):
            if current_pair == [target_pair[1], target_pair[0]]:
                ids_to_drop.append(edge_list.index(target_pair))
    edge_list = np.delete(edge_list, ids_to_drop, axis=0)

    return edge_list


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


class ECFPMolFeaturizer:

    def __init__(self, fptype=None):
        if fptype is None:
            fptype = {"Radius": 6, "Size": 512}
        self.fptype = fptype

    def _featurize(self, molecule):
        """
        :param molecule: molecule object
        :param fptype: type, radius and size of fingerprint
        :type fptype: dict
        :return: molstring for ecfp fingerprint
        """
        arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(
                molecule, self.fptype['Radius'], self.fptype['Size'], useFeatures=False
            ), arr
        )
        return torch.tensor(arr)

    def featurize_sdf(self, path_to_sdf):
        mols = Chem.SDMolSupplier(path_to_sdf)

        return [self._featurize(mol) for mol in mols if mol]


class ConvMolFeaturizer():
    name = ['conv_mol']

    def __init__(self, master_atom=False, use_chirality=False,
                 atom_properties=[]):
        """
        Parameters
        ----------
        master_atom: Boolean
          if true create a fake atom with bonds to every other atom.
          the initialization is the mean of the other atom features in
          the molecule.  This technique is briefly discussed in
          Neural Message Passing for Quantum Chemistry
          https://arxiv.org/pdf/1704.01212.pdf
        use_chirality: Boolean
          if true then make the resulting atom features aware of the
          chirality of the molecules in question
        atom_properties: list of string or None
          properties in the RDKit Mol object to use as additional
          atom-level features in the larger molecular feature.  If None,
          then no atom-level properties are used.  Properties should be in the
          RDKit mol object should be in the form
          atom XXXXXXXX NAME
          where XXXXXXXX is a zero-padded 8 digit number coresponding to the
          zero-indexed atom index of each atom and NAME is the name of the property
          provided in atom_properties.  So "atom 00000000 sasa" would be the
          name of the molecule level property in mol where the solvent
          accessible surface area of atom 0 would be stored.

        Since ConvMol is an object and not a numpy array, need to set dtype to
        object.
        """
        self.dtype = object
        self.master_atom = master_atom
        self.use_chirality = use_chirality
        self.atom_properties = list(atom_properties)

    def _get_atom_properties(self, atom):
        """
        For a given input RDKit atom return the values of the properties
        requested when initializing the featurize.  See the __init__ of the
        class for a full description of the names of the properties

        Parameters
        ----------
        atom: RDKit.rdchem.Atom
          Atom to get the properties of
        returns a numpy lists of floats of the same size as self.atom_properties
        """
        values = []
        for prop in self.atom_properties:
            mol_prop_name = str("atom %08d %s" % (atom.GetIdx(), prop))
            try:
                values.append(float(atom.GetOwningMol().GetProp(mol_prop_name)))
            #     values.extend(fukui_desc)
            except KeyError:
                raise KeyError("No property %s found in %s in %s" %
                               (mol_prop_name, atom.GetOwningMol(), self))
        return np.array(values)

    def get_fukui(self, mol) -> []:
        pass

    def _featurize(self, mol, fukui_conf=100):
        """Encodes mol as a ConvMol object."""
        # Get the node features
        idx_nodes = [(a.GetIdx(),
                      np.concatenate((atom_features(
                          a, use_chirality=self.use_chirality),
                                      self._get_atom_properties(a))))
                     for a in mol.GetAtoms()]
        # TODO add fukui indices addition here

        idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
        idx, nodes = list(zip(*idx_nodes))

        # Stack nodes into an array
        nodes = np.vstack(nodes)
        if self.master_atom:
            master_atom_features = np.expand_dims(np.mean(nodes, axis=0), axis=0)
            nodes = np.concatenate([nodes, master_atom_features], axis=0)

        # Get bond lists with reverse edges included
        edge_list = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
        ]

        # Get canonical adjacency list
        canon_adj_list = [[] for mol_id in range(len(nodes))]
        for edge in edge_list:
            canon_adj_list[edge[0]].append(edge[1])
            canon_adj_list[edge[1]].append(edge[0])

        if self.master_atom:
            fake_atom_index = len(nodes) - 1
            for index in range(len(nodes) - 1):
                canon_adj_list[index].append(fake_atom_index)

        return Data(x=torch.tensor(nodes, dtype=torch.float32),
                    edge_index=adj_to_edge(canon_adj_list).t().contiguous())

    def feature_length(self):
        return 75 + len(self.atom_properties)

    def __hash__(self):
        atom_properties = tuple(self.atom_properties)
        return hash((self.master_atom, self.use_chirality, atom_properties))

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return False
        return self.master_atom == other.master_atom and \
               self.use_chirality == other.use_chirality and \
               tuple(self.atom_properties) == tuple(other.atom_properties)


class SkipatomFeaturizer():
    """
    Class for extracting element features by skipatom_models approach

    Attributes
    ----------
    model : <class 'torch.nn.Module'>
        skipatom_models model for featurizing

    Methods
    ----------
    _featurize(element : str)
        extract element features by skipatom_models approach
    """

    def __init__(self, vectors_filename="Source/featurizers/skipatom_vectors_dim200.torch"):
        self.get_vector = torch.load(vectors_filename)

    def _featurize(self, element):
        """

        Parameters
        ----------
        element : str
            string representation of an element to be featurized

        Returns
        -------
            features : torch.tensor
                features of an element obtained from skipatom approach, shape (1, 200)
        """
        return self.get_vector[element].unsqueeze(0)


def get_target_values(path_to_data, valuenames):
    """
    Get values of all targets for molecules in .sdf file

    Parameters
    ----------
    path_to_data : str
        Path to .sdf file with data
    valuenames : list-like
         List of target value names in .sdf file

    Returns
    -------
    targets : list
       List where i-th element is a np.array with targets for i-th molecule
    """
    suppl = Chem.SDMolSupplier(path_to_data)
    mols = [x for x in suppl if x is not None]
    trg = [np.array(get_values_list(mol, valuenames), dtype=float) for mol in mols]
    return trg


def get_values_list(mol, valuenames):
    """
    Get property values of given molecule

    Parameters
    ----------
    mol : Chem.Mol object
        Given molecule
    valuenames : list-like
         Names of values with to get from Chem.Mol object

    Returns
    -------
    values : list
       List of property values for given molecule
    """
    try:
        return [float(Chem.Mol.GetProp(mol, vn)) for vn in valuenames]
    except:
        return [cat_to_num(Chem.Mol.GetProp(mol, vn)) for vn in valuenames]


def cat_to_num(value: str) -> int:
    if value.lower() == "true":
        return 1
    else:
        return 0


def featurize_sdf(path_to_data, valuenames=None, ext_test_set=False, subsample_size=None,
                  featurizer=ConvMolFeaturizer()):
    """
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_data : str
        path to .sdf file with data
    valuenames : array_like, optional
        names of target variables in .sdf file
    ext_test_set : bool, optional
        if True, returns random sample of data with size of subsample_size
    subsample_size : int, optional
        a size of returned sample if ext_test_set is True
    featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    suppl = Chem.SDMolSupplier(path_to_data)
    mols = [x for x in suppl if x is not None]
    if subsample_size and not ext_test_set:
        mols = random.sample(mols, subsample_size)

    features = [featurizer._featurize(m) for m in mols]

    if valuenames:
        trg = get_target_values(path_to_data, valuenames)
        for i, val in enumerate(trg):
            features[i].y = torch.from_numpy(val)

    return features


def featurize_sdf_with_metal(path_to_sdf,
                             mol_featurizer=ConvMolFeaturizer(),
                             metal_featurizer=SkipatomFeaturizer(),
                             data_multy_coefficients=None,
                             seed=42):
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
    data_multy_coefficients = {} if data_multy_coefficients is None else data_multy_coefficients
    suppl = Chem.SDMolSupplier(path_to_sdf)
    mols = [x for x in suppl if x is not None]
    mol_features = [mol_featurizer._featurize(m) for m in mols]

    all_data = []
    for mol_ind in range(len(mols)):
        targets = [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("logK_")]
        for target in targets:
            features = copy.deepcopy(mol_features[mol_ind])

            element_symbol = target.split("_")[-1]
            features.metal_x = metal_featurizer._featurize(element_symbol)
            features.y = torch.tensor([float(mols[mol_ind].GetProp(target))])
            if element_symbol in data_multy_coefficients:
                all_data += [features] * data_multy_coefficients[element_symbol]
            else:
                all_data += [features]
    random.Random(seed).shuffle(all_data)

    return all_data


def featurize_sdf_with_metal_and_conditions(path_to_sdf,
                                            mol_featurizer=ConvMolFeaturizer(),
                                            metal_featurizer=SkipatomFeaturizer(),
                                            data_multy_coefficients=None,
                                            seed=42):
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
    data_multy_coefficients = {} if data_multy_coefficients is None else data_multy_coefficients
    suppl = Chem.SDMolSupplier(path_to_sdf)
    mols = [x for x in suppl if x is not None]
    mol_features = [mol_featurizer._featurize(m) for m in mols]

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
            features.metal_x = torch.cat((metal_featurizer._featurize(element_symbol), condition_values), dim=-1)
            features.y = torch.tensor([logK])
            if element_symbol in data_multy_coefficients:
                all_data += [features] * data_multy_coefficients[element_symbol]
            else:
                all_data += [features]
    random.Random(seed).shuffle(all_data)

    return all_data


def featurize_smiles_np(arr, featurizer, log_every_N=1000, verbose=True):
    """Featurize individual compounds in a numpy array.

  Given a featurizer that operates on individual chemical compounds
  or macromolecules, compute & add features for that compound to the
  features array
  """
    features = []
    from rdkit.Chem import rdmolfiles
    from rdkit.Chem import rdmolops
    for ind, mol in enumerate(arr):
        if mol:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        if ind % log_every_N == 0:
            logger.debug("Featurizing sample %d" % ind, verbose)
        features.append(featurizer._featurize(mol))

    valid_inds = np.array(
        [1 if elt.size is not (0, 0) else 0 for elt in features], dtype=bool)
    features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
    features = np.squeeze(np.array(features))
    return features.reshape(-1, )


def featurize_sdf_classification(path_to_sdf, valuename="logK", mol_featurizer=ConvMolFeaturizer()):
    suppl = Chem.SDMolSupplier(path_to_sdf)
    mols = [x for x in suppl if x is not None]
    data = [mol_featurizer._featurize(m) for m in mols]
    class_ids = torch.tensor([int(m.GetProp(valuename)) for m in mols])
    # one_hots = F.one_hot(class_ids)
    # one_hots = one_hots.unsqueeze(dim=1)

    for graph, y in zip(data, class_ids):
        graph.y = y

    random.shuffle(data)
    return data
