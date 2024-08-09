import copy
import json
import random
from typing import Union, Optional

import torch
from rdkit import Chem
from rdkit.Chem import Mol

from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from config import ROOT_DIR

from rdkit.Chem import MACCSkeys, AllChem, DataStructs, Descriptors
import numpy as np


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


def featurize_sdf_with_metal(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                             metal_featurizer=SkipatomFeaturizer(),
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

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    if path_to_sdf is None and molecules is None:
        raise ValueError("'path_to_sdf' or 'molecules' parameter should be stated, got neither")
    elif path_to_sdf is not None and molecules is not None:
        raise ValueError("Only one source ('path_to_sdf' or 'molecules' parameter) should be stated, got both")
    mols = molecules or [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    mol_graphs = [mol_featurizer.featurize(m) for m in mols]

    all_data = []
    for mol, graph in zip(mols, mol_graphs):
        targets = [prop for prop in mol.GetPropNames() if prop.startswith("logK_")]
        for target in targets:
            new_graph = copy.deepcopy(graph)

            element_symbol = target.split("_")[-1]
            new_graph.metal_x = metal_featurizer.featurize(element_symbol)
            new_graph.y = {"logK": torch.tensor([[float(mol.GetProp(target))]])}
            all_data += [new_graph]
    random.Random(seed).shuffle(all_data)

    return all_data


def featurize_sdf_with_solvent(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                               seed=42, shuffle=True):
    """"
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    molecules: List[Chem.Mol]
        list of molecules, can be used instead of path_to_sdf
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    seed: int = 42
        random seed for reproducibility
    shuffle: bool
        whether to shuffle data after featurization

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    if path_to_sdf is None and molecules is None:
        raise ValueError("'path_to_sdf' or 'molecules' parameter should be stated, got neither")
    elif path_to_sdf is not None and molecules is not None:
        raise ValueError("Only one source ('path_to_sdf' or 'molecules' parameter) should be stated, got both")
    mols = molecules or [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    mol_features = [mol_featurizer.featurize(m) for m in mols]
    all_data = []
    for mol_ind in range(len(mols)):
        logS_list = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("Solubility")]:
            logS_list += [float(mols[mol_ind].GetProp(target))]
        features = copy.deepcopy(mol_features[mol_ind])
        features.x_fully_connected = torch.tensor([81.0]).unsqueeze(0)
        features.y = {"Solubility": torch.tensor([logS_list])}
        all_data += [features]

    if shuffle: random.Random(seed).shuffle(all_data)

    return all_data


class Complex:
    def __init__(self, mol: Union[str, Mol], metal: str,
                 valence: int, temperature: float, ionic_str: float,
                 logk: Optional[float] = None, use_conds=True):

        self.metal = metal
        self.valence = valence if valence else 3
        self.temperature = temperature if temperature else 20
        self.ionic_str = ionic_str if ionic_str else 0.1
        self.logk = logk
        self.use_conds = use_conds
        if isinstance(mol, str):
            self.mol = Chem.MolFromSmiles(mol)
        elif isinstance(mol, Mol):
            self.mol = mol
        else:
            raise ValueError(f"invalid molecule input type: {type(mol)}")

        self.mol_featurizer = ConvMolFeaturizer()
        self.metal_featurizer = SkipatomFeaturizer()

        self.graph = self.mol_featurizer.featurize(self.mol)
        conditions = torch.tensor([[self.valence, self.temperature, self.ionic_str]])
        if self.use_conds:
            self.graph.metal_x = torch.cat((self.metal_featurizer.featurize(self.metal), conditions), dim=-1)
        else:
            self.graph.metal_x = self.metal_featurizer.featurize(self.metal)

        if self.logk:
            self.graph.y = torch.tensor([self.logk])

def get_solvent_vector(str):
    """"
    Implements a vector representation of the solvent
    based on MACCS keys and rdkit descriptors

    """
    #str = mol.GetProp('Solvent')
    names_dict = {'acetyl acetate': 'CC(=O)OC(C)=O', 'chloroform': 'ClC(Cl)Cl', 'n-butanol': 'CCCCO',
                  'n-heptane': 'CCCCCCC', 'γ-butyrolactone': 'O=C1CCCO1', 'DMAC': 'CN(C)C(C)=O',
                  'isooctanol': 'CC(C)CCCCCO', 'isopropyl acetate': 'CC(C)OC(C)=O', 'n-pentanol': 'CCCCCO',
                  'ethyl formate': 'CCOC=O', 'acetylacetone': 'CC(=O)CC(C)=O', 'n-decanol': 'CCCCCCCCCCO',
                  'trichloroethylene': 'ClC=C(Cl)Cl', '1,1-dichloroethane': 'CC(Cl)Cl', 'n-propanol': 'CCCO',
                  'p-tert-butyltoluene': 'Cc1ccc(cc1)C(C)(C)C', '1,4-dioxane': 'C1COCCO1', 'acetone': 'CC(C)=O',
                  'anisole': 'COc1ccccc1', '1,2-dichlorobenzene': 'Clc1ccccc1Cl', 'tetrachloromethane': 'ClC(Cl)(Cl)Cl',
                  'n-heptanol': 'CCCCCCCO', 'sulfolane': 'O=S]1(=O)CCCC1', 'epichlorohydrin': 'ClCC1CO1',
                  'methylcyclohexane': 'CC1CCCCC1', '2-aminoethanol': 'NCCO', 'sec-butanol': 'CCC(C)O',
                  'tributyl phosphate': 'CCCCO[P(OCCCC)OCCCC', '3,6-dioxa-1-decanol': 'CCCCOCCOCCO',
                  '2-hexanone': 'CCCCC(C)=O', 'n-hexanol': 'CCCCCCO', 'pyridine': 'c1ccncc1', 'n-octane': 'CCCCCCCC',
                  'benzylalcohol': 'OCc1ccccc1', 'MIBK': 'CC(C)CC(C)=O', 'aniline': 'Nc1ccccc1', 'ethanol': 'CCO',
                  'n-pentane': 'CCCCC', 'morpholine-4-carbaldehyde': 'O=CN1CCOCC1', '2-butanone': 'CCC(C)=O',
                  'DEF': 'Fe]|1|2|3|4|5|ON(C)C(=O|1)CCC(=O)NCCCCCN(O|2)C(=O|3)CCC(=O)NCCCCCN(O|4)C(=O|5)C',
                  'NMP': 'CN1CCCC1=O', 'decalin': 'C1CCC2CCCCC2C1', '2-pentanol': 'CCCC(C)O', 'transcutol': 'CCOCCOCCO',
                  'sec-butyl acetate': 'CCC(C)OC(C)=O', '1-methoxy-2-propyl acetate': 'COCC(C)OC(C)=O',
                  '2-(2-methoxypropoxy) propanol': 'COC(C)COC(C)CO', 'cyclohexane': 'C1CCCCC1',
                  'triethyl phosphate': 'CCO[P(OCC)OCC', 'DMF': 'CN(C)C=O', 'methyl acetate': 'COC(C)=O',
                  'dichloromethane': 'ClCCl', 'tert-butanol': 'CC(C)(C)O', 'acetic acid': 'CC(O)=O',
                  'ethylene glycol': 'OCCO', 'acetonitrile': 'CC#N', 'DMSO': 'CS=O', '2-propoxyethanol': 'CCCOCCO',
                  'n-dodecane': 'CCCCCCCCCCCC', 'formamide': 'NC=O', 'THF': 'C1CCOC1', 'methanol': 'CO',
                  'isopropanol': 'CC(C)O', 'formic acid': 'OC=O', 'water': 'O', 'propionic acid': 'CCC(O)=O',
                  'dipropyl ether': 'CCCOCCC', 'diisopropyl ether': 'CC(C)OC(C)C', '2-methoxyethanol': 'COCCO',
                  'ethyl acetate': 'CCOC(C)=O', 'benzene': 'c1ccccc1', 'p-xylene': 'Cc1ccc(C)cc1',
                  '2-octanol': 'CCCCCCC(C)O', 'MTBE': 'COC(C)(C)C', '1,2-dichloroethane': 'ClCCCl',
                  'ethyl benzene': 'CCc1ccccc1', '2-ethoxyethanol': 'CCOCCO', 'n-propyl acetate': 'CCCOC(C)=O',
                  'methyl propionate': 'CCC(=O)OC', 'triethyl orthoformate': 'CCOC(OCC)OCC',
                  '1-propoxy-2-propanol': 'CCCOCC(C)O', 'diacetone alcohol': 'CC(=O)CC(C)(C)O',
                  'benzyl alcohol': 'OCc1ccccc1', '2-butoxyethanol': 'CCCCOCCO', 'propylene carbonate': 'CC1COC(=O)O1',
                  'nonan-1-ol': 'CCCCCCCCCO', 'sec-pentanol': 'CCCC(C)O', '2-isopropoxyethanol': 'CC(C)OCCO',
                  'DMS': 'CS=O', 'diethylene glycol': 'OCCOCCO', 'MEK': 'CCC(C)=O', 'n-pentyl acetate': 'CCCCCOC(C)=O',
                  'cyclohexanone': 'O=C1CCCCC1', 'tert-amyl alcohol': 'CCC(C)(C)O', 'cyclopentanone': 'O=C1CCCC1',
                  'morpholine': 'C1COCCN1', 'm-xylene': 'Cc1cccc(C)c1', '3-pentanone': 'CCC(=O)CC',
                  '2-ethylhexanol': 'CCCCC(CC)CO', 'n-hexane': 'CCCCCC', 'isooctane': 'CCCCCC(C)C',
                  'isobutanol': 'CC(C)CO', 'isopentanol': 'CC(C)CCO', 'n-hexadecane': 'CCCCCCCCCCCCCCCC',
                  'diethyl ether': 'CCOCC', 'ethylbenzene': 'CCc1ccccc1', 'n-butyric acid': 'CCCC(O)=O',
                  'acrylic acid': 'OC(=O)C=C', '2-ethyl-n-hexanol': 'CCCCC(CC)CO', 'chlorobenzene': 'Clc1ccccc1',
                  '3-oxa-1,5-pentanediol': 'OCCOCCO', '1,3-propanediol': 'OCCCO', 'DMA': 'CC(C)=CCOP(=O)OP(O)=O',
                  'n-dodecanol': 'CCCCCCCCCCCCO', 'n-butyl acetate': 'CCCCOC(C)=O',
                  'methyl 4-tert-butylbenzoate': 'COC(=O)c1ccc(cc1)C(C)(C)C', 'isobutyl acetate': 'CC(C)COC(C)=O',
                  'o-xylene': 'Cc1ccccc1C', '1-methoxy-2-propanol': 'COCC(C)O',
                  '1,2,4-trichlorobenzene': 'Clc1ccc(Cl)c(Cl)c1', 'toluene': 'Cc1ccccc1', 'methyl formate': 'COC=O',
                  'tert-butyl acetate': 'CC(=O)OC(C)(C)C', 'n-octanol': 'CCCCCCCCO', 'propylene glycol': 'CC(O)CO',
                  '4-methylpyridine': 'Cc1ccncc1', 'acetophenone': 'CC(=O)c1ccccc1', '2-pentanone': 'CCCC(C)=O',
                  '1-bromopropane': 'CCCBr', 'p-cymene': 'CC(C)c1ccc(C)cc1', 'cumene': 'CC(C)c1ccccc1',
                  '2-(2-butoxyethoxy)ethanol': 'CCCCOCCOCCO', '1,2-diethoxyethane': 'CCOCCOCC'}

    solvent_mol = Chem.MolFromSmiles(names_dict[str])
    if solvent_mol is not None:
        vector = []

        vector += [Chem.Descriptors.MaxAbsPartialCharge(solvent_mol)]
        vector += [Chem.EState.EState.MaxAbsEStateIndex(solvent_mol)]
        vector += [Chem.Descriptors.MinPartialCharge(solvent_mol)]
        vector += [Chem.EState.EState.MinAbsEStateIndex(solvent_mol)]
        vector += [Chem.Descriptors.MaxPartialCharge(solvent_mol)]
        vector += [Chem.EState.EState.MinEStateIndex(solvent_mol)]
        vector += [Chem.Descriptors.MinAbsPartialCharge(solvent_mol)]
        vector += [Chem.EState.EState.MaxEStateIndex(solvent_mol)]
        vector += [Chem.Crippen.MolMR(solvent_mol)]

        vector += [Chem.MolSurf.SlogP_VSA2(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState2(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA1(solvent_mol)]
        vector += [Chem.EState.EState_VSA.EState_VSA1(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA1(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState8(solvent_mol)]
        vector += [Chem.EState.EState_VSA.EState_VSA9(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState6(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA3(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState1(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState3(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA8(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA5(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA7(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA7(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState9(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA6(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA1(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA3(solvent_mol)]
        vector += [Chem.MolSurf.SMR_VSA10(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA5(solvent_mol)]
        vector += [Chem.EState.EState_VSA.EState_VSA5(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA2(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA4(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA6(solvent_mol)]
        vector += [Chem.EState.EState_VSA.EState_VSA8(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA12(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA14(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState10(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA13(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA12(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA6(solvent_mol)]
        vector += [Chem.EState.EState_VSA.VSA_EState4(solvent_mol)]
        vector += [Chem.MolSurf.PEOE_VSA9(solvent_mol)]
        vector += [Chem.MolSurf.SlogP_VSA10(solvent_mol)]
        vector += [Chem.MolSurf.TPSA(solvent_mol)]

        vector += [Chem.GraphDescriptors.Chi1(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi1v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi1n(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi4v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi2v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi3v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi4n(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi0n(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi0(solvent_mol)]
        vector += [Chem.GraphDescriptors.Kappa1(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi0v(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi3n(solvent_mol)]
        vector += [Chem.GraphDescriptors.Kappa3(solvent_mol)]
        vector += [Chem.GraphDescriptors.Kappa2(solvent_mol)]
        vector += [Chem.GraphDescriptors.Chi2n(solvent_mol)]
        vector += [Chem.GraphDescriptors.BalabanJ(solvent_mol)]
        vector += [Chem.GraphDescriptors.HallKierAlpha(solvent_mol)]
        vector += [Chem.GraphDescriptors.BertzCT(solvent_mol)]
        vector += [Chem.Lipinski.HeavyAtomCount(solvent_mol)]
        vector += [Chem.Descriptors.MolWt(solvent_mol)]
        vector += [Chem.Lipinski.NHOHCount(solvent_mol)]
        vector += [Chem.Lipinski.NOCount(solvent_mol)]
        vector += [Chem.Crippen.MolLogP(solvent_mol)]
        vector += [Chem.Lipinski.NumHDonors(solvent_mol)]
        vector += [Chem.Lipinski.NumHeteroatoms(solvent_mol)]
        vector += [Chem.Lipinski.NumHAcceptors(solvent_mol)]
        vector += [Chem.Lipinski.NumRotatableBonds(solvent_mol)]
        vector += [Chem.Descriptors.NumValenceElectrons(solvent_mol)]
        vector += [Chem.Lipinski.NumAromaticCarbocycles(solvent_mol)]
        vector += [Chem.Lipinski.NumAromaticRings(solvent_mol)]
        vector += [Chem.Lipinski.NumAromaticHeterocycles(solvent_mol)]
        vector += [Chem.Lipinski.NumSaturatedHeterocycles(solvent_mol)]

        vector += AllChem.GetMACCSKeysFingerprint(solvent_mol)

        #vector += [float(mol.GetProp('T,K'))]
        # return torch.tensor(vector).unsqueeze(0)
        return vector
    else:
        raise ValueError(f"invalid molecule input type: {mol.GetProp('Solvent')}")
def featurize_sdf_with_solvent_elina(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                               seed=42, shuffle=True):
    """"
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    molecules: List[Chem.Mol]
        list of molecules, can be used instead of path_to_sdf
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    seed: int = 42
        random seed for reproducibility
    shuffle: bool
        whether to shuffle data after featurization

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    if path_to_sdf is None and molecules is None:
        raise ValueError("'path_to_sdf' or 'molecules' parameter should be stated, got neither")
    elif path_to_sdf is not None and molecules is not None:
        raise ValueError("Only one source ('path_to_sdf' or 'molecules' parameter) should be stated, got both")
    mols = molecules or [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    mol_features = [mol_featurizer.featurize(m) for m in mols]

    all_data = []


    for mol_ind in range(len(mols)):
        logS_list = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("Solubility")]:
            logS_list += [float(mols[mol_ind].GetProp(target))]
            element_symbol = mols[mol_ind].GetProp('Solvent')
            vector = [float(mols[mol_ind].GetProp('T,K'))] + get_solvent_vector(element_symbol)
            vector = np.array(vector, dtype=np.float32)

        features = copy.deepcopy(mol_features[mol_ind])
        # features.x_fully_connected = torch.cat((get_solvent_vector(element_symbol), conditions), dim=-1)
        features.x_fully_connected = torch.tensor(vector).unsqueeze(0)
        features.y = {"Solubility": torch.tensor([logS_list])}
        all_data += [features]

    if shuffle: random.Random(seed).shuffle(all_data)


    return all_data


def featurize_sdf_with_solvent_elina_MOPAC(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                               seed=42, shuffle=True):
    """"
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    molecules: List[Chem.Mol]
        list of molecules, can be used instead of path_to_sdf
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    seed: int = 42
        random seed for reproducibility
    shuffle: bool
        whether to shuffle data after featurization

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    if path_to_sdf is None and molecules is None:
        raise ValueError("'path_to_sdf' or 'molecules' parameter should be stated, got neither")
    elif path_to_sdf is not None and molecules is not None:
        raise ValueError("Only one source ('path_to_sdf' or 'molecules' parameter) should be stated, got both")
    mols = molecules or [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    mol_features = [mol_featurizer.featurize(m) for m in mols]

    all_data = []


    for mol_ind in range(len(mols)):
        logS_list = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("Solubility")]:
            logS_list += [float(mols[mol_ind].GetProp(target))]
            element_symbol = mols[mol_ind].GetProp('Solvent')
            vector = [float(mols[mol_ind].GetProp('T,K'))] + get_solvent_vector(element_symbol)
            vector += [float(mols[mol_ind].GetProp('dGsolv'))]
            vector += [float(mols[mol_ind].GetProp('eps'))]
            vector += [float(mols[mol_ind].GetProp('BP'))]
            vector = np.array(vector, dtype=np.float32)

        features = copy.deepcopy(mol_features[mol_ind])
        # features.x_fully_connected = torch.cat((get_solvent_vector(element_symbol), conditions), dim=-1)
        features.x_fully_connected = torch.tensor(vector).unsqueeze(0)
        features.y = {"Solubility": torch.tensor([logS_list])}
        all_data += [features]

    if shuffle: random.Random(seed).shuffle(all_data)


    return all_data


def featurize_sdf_with_solvent_MP_BP(path_to_sdf=None, molecules=None, mol_featurizer=ConvMolFeaturizer(),
                               seed=42, shuffle=True):
    """"
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    molecules: List[Chem.Mol]
        list of molecules, can be used instead of path_to_sdf
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule
    seed: int = 42
        random seed for reproducibility
    shuffle: bool
        whether to shuffle data after featurization

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    if path_to_sdf is None and molecules is None:
        raise ValueError("'path_to_sdf' or 'molecules' parameter should be stated, got neither")
    elif path_to_sdf is not None and molecules is not None:
        raise ValueError("Only one source ('path_to_sdf' or 'molecules' parameter) should be stated, got both")
    mols = molecules or [mol for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None]
    mol_features = [mol_featurizer.featurize(m) for m in mols]

    all_data = []


    for mol_ind in range(len(mols)):
        logS_list = []
        for target in [prop for prop in mols[mol_ind].GetPropNames() if prop.startswith("Solubility")]:
            logS_list += [float(mols[mol_ind].GetProp(target))]
            #element_symbol = mols[mol_ind].GetProp('Solvent')
            vector = [float(mols[mol_ind].GetProp('BP'))] #+ get_solvent_vector(element_symbol)
            #vector += [float(mols[mol_ind].GetProp('MP'))]
            vector = np.array(vector, dtype=np.float32)

        features = copy.deepcopy(mol_features[mol_ind])
        # features.x_fully_connected = torch.cat((get_solvent_vector(element_symbol), conditions), dim=-1)
        features.x_fully_connected = torch.tensor(vector).unsqueeze(0)
        features.y = {"Solubility": torch.tensor([logS_list])}
        all_data += [features]

    if shuffle: random.Random(seed).shuffle(all_data)


    return all_data