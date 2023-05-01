import random
import re

import torch
from rdkit import Chem
import pandas as pd
import os
from rdkit.Chem import Draw

MODEL = torch.load("Source/featurizers/skipatom_vectors_dim200.torch")


def metal_valid(metal):
    return metal in MODEL


def get_logK_and_metal(molecule):
    """
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol object

    Returns
    -------
    None if molecule has not property like "logK_{metal}"
    LogK : float
        log of stability constant
    metal : str
        symbol of metal element
    """
    if molecule is None:
        return None
    ans = []
    for PropName in list(molecule.GetPropNames()):
        if PropName.startswith("logK"):
            logK = float(molecule.GetProp(PropName))
            metal = PropName.split("_")[-1]
            ans += [(logK, metal)]
    return ans


def save_to_sdf(list_of_molecules, output_path):
    """
    Parameters
    ----------
    list_of_molecules : list
        list of rdkit.Chem.rdchem.Mol objects to be saved
    output_path: str
        path to output .sdf file
    """
    writer = Chem.SDWriter(output_path)
    for molecule in list_of_molecules:
        writer.write(molecule)


def are_molecules_same(mol1, mol2):
    """
    Parameters
    ----------
    mol1 : rdkit.Chem.rdchem.Mol object
    mol2 : rdkit.Chem.rdchem.Mol object

    Returns
    -------
    are_same : bool
        True if molecules are the same
        False if molecules differ
    """
    # TODO: change the check to distinguish stereoisomers
    return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)


def read_sdf(path):
    try:
        result = [mol for mol in Chem.SDMolSupplier(path) if mol is not None]
    except OSError:
        return []
    return result


def save_mols_to_png(molecules, output_path):
    """
    Save molecules structures as .png image

    Parameters
    ----------
    molecules : list of rdkit.Chem.rdchem.Mol objects
        list of molecules to be saved
    output_path : str
        path to output .png file
    """
    img = Draw.MolsToGridImage(molecules, molsPerRow=2, subImgSize=(200, 200),
                               legends=[
                                   molecule.GetProp("Name") if "Name" in list(molecule.GetPropNames()) else "Unknown"
                                   for molecule in molecules
                               ])
    img.save(output_path)


def join_sdf(input_paths, output_path=None):
    """
    Join several .sdf files and rename property "logK" => "logK_{metal}"

    Parameters
    ----------
    input_paths : dict
        keys are metal symbols and values are paths to .sdf files to be joined
    output_path : str
        path to output .sdf file

    Returns
    ----------
    all_molecules : list
        list of all molecules in joined .sdf files
    """
    all_molecules = []
    for metal in input_paths:
        path = input_paths[metal]
        for molecule in Chem.SDMolSupplier(path):
            if molecule is not None and "LogK" in list(molecule.GetPropNames()):
                logK = molecule.GetProp("LogK")
                molecule.ClearProp("LogK")
                molecule.SetProp(f"logK_{metal}", logK)
                all_molecules += [molecule]
    if output_path:
        save_to_sdf(all_molecules, output_path)
    return all_molecules


def concat_sdf(input_paths, output_path):
    """
    Concatenate several .sdf files

    Parameters
    ----------
    input_paths : list
        list of paths to .sdf files to be concatenated
    output_path : str
        path to output .sdf file
    """
    all_molecules = [molecule for path in input_paths
                     for molecule in Chem.SDMolSupplier(path)
                     if molecule is not None]
    save_to_sdf(all_molecules, output_path)


def concat_csv(input_paths, output_path):
    """
    Concatenate several .csv files with same column names

    Parameters
    ----------
    input_paths : list
        list of paths to .csv files to be concatenated
    output_path : str
        path to output .scv file
    """
    dfs = []
    for path in input_paths:
        dfs += [pd.read_csv(path)]
    output_df = pd.concat(dfs, ignore_index=True)
    output_df.to_csv(output_path)


def read_csv(input_path, output_path=None):
    """
    Transform .csv file to .sdf with properties like "logK_{metal}"

    Parameters
    ----------
    input_path : str
        path to .csv file to be transformed into .sdf
    output_path : str
        path to output .sdf file

    Returns
    ----------
    all_molecules : list
        list of all molecules from .csv
    """
    df = pd.read_csv(input_path)
    all_molecules = []
    for i in df.index:
        metal = df.loc[i, "metal"]
        mol = Chem.MolFromSmiles(df.loc[i, "smiles"])
        if mol is None:
            continue
        mol.SetProp("Name", df.loc[i, "name"])
        mol.SetProp(f"logK_{metal}", str(df.loc[i, "constant"]))
        all_molecules += [mol]
    if output_path:
        save_to_sdf(all_molecules, output_path)
    return all_molecules


def join_same_mol_in_sdf(molecules, output_path=None):
    """
    Remove repetitions of the same molecule in .sdf file and write all "logK_{metal}" properties to one molecule

    Parameters
    ----------
    molecules : list
        list of molecules with repetitions of the same molecule for different metals
    output_path : str, optional
        path to output .sdf file without repetitions of the same molecule

    Returns
    ----------
    result : list
        list of molecules without repetitions of the same molecule
    """
    result = []
    for molecule in molecules:
        for logK, metal in get_logK_and_metal(molecule):
            mol_exists = False
            for i in range(len(result)):
                if are_molecules_same(molecule, result[i]):
                    # save_mols_to_png((molecule, all_molecules[i]), output_path=f"Data/imgs/{metal}_{i}.png")
                    result[i].SetProp(f"logK_{metal}", str(logK))
                    mol_exists = True
                    break

            if not mol_exists:
                result += [molecule]
    if output_path:
        save_to_sdf(result, output_path)
    return result


def train_test_split_sdf(path_to_sdf, test_ratio=0.1):
    """
    Select a test dataset balanced by different metals

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with all data
    test_ratio : float
        percentage of test data

    Returns
    -------
    train_dataset : list
        list of molecules to be used in training model
    test_dataset: list
        list of molecules to be used in testing model
    """
    mols = [x for x in Chem.SDMolSupplier(path_to_sdf) if x is not None]
    train_data = []
    test_data = []

    all_molecules = {}
    for mol in mols:
        targets = [prop for prop in mol.GetPropNames() if prop.startswith("logK_")]
        for target in targets:
            element_symbol = target.split("_")[-1]
            if element_symbol in all_molecules:
                all_molecules[element_symbol] += [mol]
            else:
                all_molecules[element_symbol] = [mol]
    test_size = int(sum([len(l) for l in all_molecules.values()]) * test_ratio)
    num_of_samples = test_size // len(all_molecules.keys())
    for element_symbol in all_molecules:
        current_num_of_samples = min(num_of_samples, len(all_molecules[element_symbol]) - 1)
        random.shuffle(all_molecules[element_symbol])
        test_data += all_molecules[element_symbol][:current_num_of_samples]
        train_data += all_molecules[element_symbol][current_num_of_samples:]
    return train_data, test_data


# if __name__ == "__main__":
#     model = SkipAtomInducedModel.load(
#         "skipatom_models/data/mp_2020_10_09.dim200.model",
#         "skipatom_models/data/mp_2020_10_09.training.data",
#         min_count=2e7, top_n=5)
#
#     metal_ds_folder = "Data/nist46/metal_datasets"
#     all_csv_paths = []
#     for folder_name in os.listdir(metal_ds_folder):
#         if os.path.isdir(os.path.join(metal_ds_folder, folder_name)):
#             for file_name in os.listdir(os.path.join(metal_ds_folder, folder_name)):
#                 if file_name.endswith("_constants_all.csv"):
#                     metal = file_name.split("_")[0]
#                     if metal in model.dictionary:
#                         all_csv_paths += [os.path.join(metal_ds_folder, folder_name, file_name)]
#
#     metals_list = ["Ce", "Dy", "Er", "Eu", "Gd", "Ho", "La", "Lu", "Nd", "Pr", "Sm", "Tb", "Tm", "Yb", "Y"]
#     sdf_paths = {metal: f"Data/metals/{metal}_dataset.sdf" for metal in metals_list if metal in model.dictionary}
#
#     join_sdf(sdf_paths, output_path="Data/tmp/joined_metals.sdf")
#
#     concat_csv(all_csv_paths, "Data/tmp/all_nist_metals.csv")
#     csv_to_sdf("Data/tmp/all_nist_metals.csv", output_path="Data/tmp/all_nist_metals.sdf")
#
#     concat_sdf(("Data/tmp/joined_metals.sdf", "Data/tmp/all_nist_metals.sdf"), output_path="Data/tmp/all_metals.sdf")
#     join_same_mol_in_sdf("Data/tmp/all_metals.sdf", "Data/logK_metals.sdf")
#
#     train_data, test_data = train_test_split_sdf("Data/logK_metals.sdf", test_ratio=0.02)
#     save_to_sdf(train_data, output_path="Data/logK_metals_train.sdf")
#     save_to_sdf(test_data, output_path="Data/logK_metals_test.sdf")

if __name__ == "__main__":
    files = [f for f in os.listdir("Data/nist46/") if re.fullmatch(r"\w*_ML.sdf", f)]
    all_molecules = []
    for filename in files:
        metal = filename.split("_")[0]
        # metal = re.sub(r"\d+", "", metal)
        if not metal_valid(metal):
            print(metal)
            continue
        for mol in read_sdf(os.path.join("Data/nist46/", filename)):
            if "logK" in list(mol.GetPropNames()):
                logK = mol.GetProp("logK")
                mol.ClearProp("logK")
                mol.SetProp(f"logK_{metal}", logK)
                all_molecules += [mol]
    join_same_mol_in_sdf(all_molecules, output_path="Data/logK_metals.sdf")
