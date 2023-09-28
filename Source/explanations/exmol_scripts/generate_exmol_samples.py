import os.path
import pickle
import sys
from argparse import ArgumentParser

import exmol
from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn
from tqdm.auto import tqdm

sys.path.append(os.path.abspath("."))

from config import ROOT_DIR
from Source.explanations.exmol_explanations import ModelExmol
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.trainer import ModelShell
from Source.data import root_mean_squared_error


def get_samples(smiles, metal, z, T, I):
    filename = f"{metal}_T={T}_I={I}_z={z}_{len(smiles)}.pkl"

    if not os.path.exists(os.path.join(save_folder, filename)):
        model = ModelShell(GCNN_FCNN, train_folder)
        model_exmol = ModelExmol(model, metal=metal, charge=z, temperature=T, ionic_str=I)
        samples = exmol.sample_space(smiles, model_exmol, batched=False, use_selfies=True, quiet=True,
                                     num_samples=num_samples)

        with open(os.path.join(save_folder, filename), "wb") as file:
            pickle.dump(samples, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num-processes', default=1, type=int, help='Number of processes to use. Default 1.')
    args = parser.parse_args()

    cv_folds = 5
    seed = 23
    batch_size = 64
    epochs = 1000
    es_patience = 100
    train_sdf_folder = ROOT_DIR / "Data/OneM_cond_adds"
    train_folder = ROOT_DIR / "Output/WithCondAdd/5fold/Ac_5fold_regression_2023_05_30_21_09_41"
    save_folder = ROOT_DIR / "Output/exmol_samples_3k_patterns"
    os.makedirs(save_folder, exist_ok=True)

    targets = ({
                   "name": "logK",
                   "mode": "regression",
                   "dim": 1,
                   "metrics": {
                       "R2": (r2_score, {}),
                       "RMSE": (root_mean_squared_error, {}),
                       "MAE": (mean_absolute_error, {})
                   },
                   "loss": nn.MSELoss(),
               },)

    patterns = {
        "ether": ["O=C(O)COCC(=O)O", "CC(OC(C)C(=O)O)C(=O)O", "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC",
                  "CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@H](C)C(=O)N(CCCCCCCC)CCCCCCCC",
                  "CCCCCCCCN(CCCCCCCC)C(=O)CO[C@@H](C)C(=O)N(CCCCCCCC)CCCCCCCC", "O=C(O)C1OC(C(=O)O)C(C(=O)O)C1C(=O)O"],
        "biPy": ["CCN(C(=O)c1cccc(-c2cccc(C(=O)N(CC)c3ccccc3)n2)n1)c1ccccc1",
                 "c1ccc(-c2ccc3ccc4ccc(-c5ccccn5)nc4c3n2)nc1",
                 "CCCCCCCCN(CCCCCCCC)C(=O)c1cccc(-c2cccc(C(=O)N(CCCCCCCC)CCCCCCCC)n2)n1",
                 "CCCCN(CCCC)C(=O)c1cccc(-c2cccc(C(=O)N(CCCC)CCCC)n2)n1", "c1ccc(-c2ccccn2)nc1",
                 "O=S(O[Na])c1cccc(-c2nnc(-c3cccc(-c4cccc(-c5nnc(-c6cccc(S(=O)O[Na])c6)c(-c6cccc(S(=O)O[Na])c6)n5)n4)n3)nc2-c2cccc(S(=O)O[Na])c2)c1",
                 "c1ccc(-c2cccc(-c3nnn[nH]3)n2)[nH+]c1",
                 "CCN(C(=O)c1cc([N+](=O)[O-])cc(-c2cc([N+](=O)[O-])cc(C(=O)N(CC)c3ccccc3)n2)n1)c1ccccc1",
                 "CCN(C(=O)c1cccc(-c2cccc(C(=O)N(CC)c3ccc(F)cc3)n2)n1)c1ccc(F)cc1",
                 "CCN(C(=O)c1cccc(-c2cccc(C(=O)N(CC)c3cccc(F)c3)n2)n1)c1cccc(F)c1",
                 "CCCCCc1nnc(-c2cccc(-c3ccccn3)n2)nc1CCCCC",
                 "CC1(C)CCC(C)(C)c2nc(-c3cccc(-c4cccc(-c5nnc6c(n5)C(C)(C)CCC6(C)C)n4)n3)nnc21",
                 "CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC"],
        "phenanthroline": ["c1ccc(-c2ccc3ccc4ccc(-c5ccccn5)nc4c3n2)nc1", "OCc1ccc2ccc3ccc(CO)nc3c2n1",
                           "NC(=O)c1ccc2ccc3ccc(C(N)=O)nc3c2n1",
                           "c1ccc2nc(-c3ccc4ccc5ccc(-c6ncc7ccccc7n6)nc5c4n3)ncc2c1",
                           "CC(C)(C)NCc1ccc2ccc3ccc(CNC(C)(C)C)nc3c2n1"],
        "Py": ["O=C(O)c1cccc(C(=O)O)n1", "CC(C)N(C(=O)c1cccc(C(=O)N(C(C)C)C(C)C)n1)C(C)C",
               "O=C(O)c1cccc(CN2CCOCCOCCN(Cc3cccc(C(=O)O)n3)CCOCCOCC2)n1", "Cc1cccc(CN(CC(=O)O)CC(=O)O)n1",
               "CCN(CC)C(=O)c1cccc(C(=O)N(CC)CC)n1", "O=C(c1cccc(C(=O)N(Cc2ccccc2)Cc2ccccc2)n1)N(Cc1ccccc1)Cc1ccccc1",
               "O=C(O)CN1CCCN(CC(=O)O)Cc2cccc(n2)CN(CC(=O)O)CCC1",
               "O=C(O)c1cccc(CN(CCN(Cc2cccc(C(=O)O)n2)Cc2cccc(C(=O)O)n2)Cc2cccc(C(=O)O)n2)n1",
               "O=P(O)(O)CN(Cc1cccc(CN(CP(=O)(O)O)CP(=O)(O)O)n1)CP(=O)(O)O", "Nc1cc(C(=O)O)nc(C(=O)O)c1",
               "C1=CC(c2ncc3ccccc3n2)=NC2C1=CCc1ccc(-c3ncc4ccccc4n3)nc12",
               "CCCc1nnc(-c2cccc(-c3nnc(CCC)c(CCC)n3)n2)nc1CCC",
               "Cc1cccc(C(=O)O)n1", "O=C(O)CN(CCN(CC(=O)O)c1cccc(C(=O)O)n1)Cc1cccc(C(=O)O)n1",
               "O=C(O)c1cccc(CN2CCOCCOCCN(Cc3cccc(C(=O)O)n3)CCOCC2)n1",
               "CC[C@@H](C)n1c(-c2cccc(-c3nc4ccccc4n3[C@@H](C)CC)n2)nc2ccccc21",
               "O=C(O)CN(CCCN(CC(=O)O)c1cccc(C(=O)O)n1)Cc1cccc(C(=O)O)n1",
               "O=S(=O)(O[Na])c1cccc(-c2nnc(-c3cccc(-c4nnc(-c5cccc(S(=O)(=O)O[Na])c5)c(-c5cccc(S(=O)(=O)O[Na])c5)n4)n3)nc2-c2cccc(S(=O)(=O)O[Na])c2)c1"],
        "EDTA": ["O=C(O)CN(CC(=O)O)CC(C(=O)O)N(CC(=O)O)CC(=O)O", "CC(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O",
                 "CCCCCCC(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O", "CC(C)(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O",
                 "O=C(O)CN1CCN(CC(=O)O)CC(=O)OCCOC(=O)C1", "O=C(O)CN(CC(=O)O)C1CCCCC1N(CC(=O)O)CC(=O)O",
                 "CCC(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O", "O=C(O)CN(CC(=O)O)[C@@H]1CCCC[C@H]1N(CC(=O)O)CC(=O)O",
                 "O=C(O)CN1CCN(CC(=O)O)CC(=O)OCCOCCOCCOC(=O)C1", "CC(C(C)N(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O",
                 "O=C(O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O", "O=C(O)CN1CCN(CC(=O)O)CC(=O)OCOCOCOCCOC(=O)C1",
                 "O=C(O)CN1CCNC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)NCC1", "CCCCC(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O",
                 "O=C(O)CN1CCN(CC(=O)O)CC(=O)OCCOCCOC(=O)C1", "CCCC(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O",
                 "CCCCCC[C@@H](CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O", "C[C@@H](CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O",
                 "CCCC[C@@H](CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O",
                 "O=C(O)CNC(=O)CN(CCN(CC(=O)O)CC(=O)NCC(=O)O)CC(=O)O",
                 "O=C(O)CN(CC(=O)O)C[C@@H](C(=O)O)N(CC(=O)O)CC(=O)O",
                 "O=C(O)CN(CC(=O)O)[C@@H]1CCCC[C@@H]1N(CC(=O)O)CC(=O)O",
                 "CCC[C@@H](CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O", "O=C(O)CN(CCN(CC(=O)O)C(C(=O)O)C1CCCCC1O)CC(=O)O"], }
    molecules = []
    for smiles in patterns.values():
        for s in smiles:
            if s not in molecules:
                molecules += [s]

    other_metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Mo', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs', 'Ba', 'Hf', 'Re',
                    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
    Ln_metals = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', ]
    Ac_metals = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf']

    metals = list(set(["Y", "Sc"] + Ln_metals + Ac_metals) - {"Ac", "Pa"})
    num_samples = 3000

    z = 3
    T = 25
    I = 0.1

    data = [(s, metal, z, T, I) for s in molecules for metal in metals]

    # data = []
    # folder = str(ROOT_DIR / "Data/OneM_cond_adds/")
    # for filename in tqdm(os.listdir(folder), desc="Prepare data"):
    #     for mol in Chem.SDMolSupplier(os.path.join(folder, filename)):
    #         if mol is None: continue
    #         smiles = Chem.MolToSmiles(mol)
    #         for prop_name in mol.GetPropNames():
    #             if not prop_name.startswith("logK"): continue
    #             _, metal, z, T, I = prop_name.split("_")
    #             if metal not in metals: continue
    #             z = float(z.split("=")[1])
    #             T = float(T.split("=")[1])
    #             I = float(I.split("=")[1])
    #             logK = float(mol.GetProp(prop_name))
    #
    #             data += [(smiles, metal, z, T, I, logK)]

    for params in tqdm(data, desc="Get samples"):
        get_samples(*params)
