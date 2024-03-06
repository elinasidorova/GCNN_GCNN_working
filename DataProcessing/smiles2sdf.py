import pandas as pd
from rdkit import Chem


metal = input("Enter the metal to expand sdf for: ")
path = {
    'csv': f'..Data/csv/{metal}.csv',
    'old_sdf': '..Data/sdf/{metal}.sdf',
    'new_sdf': '..Data/sdf/new_{metal}.sdf'
}

def df_tranformer(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    The function to be implemented.
    """
    return df

df = pd.read_csv(path['csv'])
df = df_transformer(df)

def get_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol

def get_props(index: int) -> dict:
    return df.iloc[index].to_dict()

def set_mol_props(mol, properties: dict):
    for prop, value in properties.items():
        prop = str(prop)
        value = str(value)
        mol.SetProp(prop, value)
    return mol


with open(path['new_sdf'], 'a') as sdf:
    # Creating the SDWriter object
    sdf_writer = Chem.SDWriter(sdf)
    
    # Creating the list of molecules already presented in sdf file
    old_mols = [mol for mol in Chem.SDMolSupplier(path['old_sdf'])]
    
    ### Writing old molecules ###
    for mol in old_mols:
        sdf_writer.write(mol)
    ###
    
    ### Writing new molecules ###
    for index in df.index:
        props = get_prop(index)
        mol = get_from_smiles(props['smiles'])
        mol_props = {
            'Name': props['name'],
            f'logK_{metal}_z=1.0_T={props["T"] - 273}_I={props["I"]}': props['logK']\
            if logK_details else 'logK': props['logK']
        }
        mol = set_mol_properties(mol, mol_props)
        sdf_writer.write(mol)

    sdf_writer.close()
    ###
