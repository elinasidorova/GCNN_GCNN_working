from rdkit import Chem
import pandas as pd


metal = input("Enter the metal to add SMILES for: ")
csv_path = f"..Data/csv/{metal}.csv"

df = pd.read_csv(csv_path)
no_smiles = df.query("smiles == '?'").name.to_list()

def add_smiles(name, smiles_string, df=df):
    df.loc[df['name'] == name, 'smiles'] = smiles_string
    return df

for name in no_smiles:
    try:
        smiles_string = input(f"SMILES for {name}: ")
    except KeyboardInterrupt:
        print(df.head())
        break

    if smiles_string:
        add_smiles(name, smiles_string)
    else:
        print('Skipped!')
