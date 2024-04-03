import json
from typing import Union, Optional

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Mol

from Source.models.GCNN.featurizers import featurize_df as featurize_df_GCNN
from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from config import ROOT_DIR


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
        return torch.tensor(self.get_vector[element], dtype=torch.float32).unsqueeze(0)


def featurize_df(df: pd.DataFrame, mol_featurizer, metal_featurizer, target="logK", conditions=()):
    graphs = featurize_df_GCNN(df, mol_featurizer, target=target)
    for i, graph in enumerate(graphs):
        graph.metal_x = torch.cat((
            metal_featurizer.featurize(df["metal"][i]),
            torch.tensor([[df[cond][i] for cond in conditions]], dtype=torch.float32)
        ), dim=-1)

    return graphs


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
