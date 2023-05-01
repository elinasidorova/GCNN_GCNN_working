import torch
from rdkit import Chem
from rdkit.Chem import Draw

MODEL = torch.load("Source/featurizers/skipatom_vectors_dim200.torch")


def metal_valid(metal):
    return metal in MODEL


def create_sdf(input_path, output_path, metals_list, property_name="LogK"):
    all_molecules = []
    for metal in metals_list:
        for molecule in Chem.SDMolSupplier(f"{input_path}/{metal}_ML.sdf"):
            if molecule is not None:
                mol_exists = False
                for i in range(len(all_molecules)):
                    # TODO: change the check to distinguish stereoisomers
                    if molecule.HasSubstructMatch(all_molecules[i]) and all_molecules[i].HasSubstructMatch(molecule):
                        # img = Draw.MolsToGridImage([molecule, all_molecules[i]], molsPerRow=2, subImgSize=(200, 200),
                        #                            legends=[
                        #                                molecule.GetProp("Name") if "Name" in list(
                        #                                    molecule.GetPropNames()) else "Unknown",
                        #                                all_molecules[i].GetProp("Name") if "Name" in list(
                        #                                    all_molecules[i].GetPropNames()) else "Unknown"
                        #                            ])
                        # img.save(f'Data/imgs/{metal}_{i}.png')
                        try:
                            logK = molecule.GetProp(property_name)
                        except KeyError:
                            raise TypeError(f"\"{property_name}\" not in {list(molecule.GetPropNames())}")
                        all_molecules[i].SetProp(f"logK_{metal}", logK)
                        mol_exists = True
                        break
                if not mol_exists:
                    try:
                        logK = molecule.GetProp(property_name)
                    except KeyError:
                        raise TypeError(f"\"{property_name}\" not in {list(molecule.GetPropNames())}")
                    molecule.ClearProp(property_name)
                    molecule.SetProp(f"logK_{metal}", logK)
                    all_molecules += [molecule]
        print(f"{metal}_ML.sdf patched")

    writer = Chem.SDWriter(output_path)
    for molecule in all_molecules:
        writer.write(molecule)


transition_metals = ["Mg", "Al", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Ag", "Cd", "Hg", "Pb", "Bi"]
Ln_metals = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Ac_metals = ["Th", "Am", "Cm", "Bk", "Cf"] # "U", "Pu",
all_metals = transition_metals + Ln_metals + Ac_metals

# for metal in all_metals:
#     if not metal_valid(metal):
#         print(f"invalid metal: {metal}!")

for metal in all_metals:
    create_sdf(
        input_path="Data/TransLnAc/",
        output_path=f"Data/OneMetal/{metal}.sdf",
        metals_list=[metal],
        property_name="logK",
    )

    # create_sdf(
    #     input_path="Data/TransLnAc/",
    #     output_path=f"Data/StrongTestonly/{metal}_testonly_test.sdf",
    #     metals_list=[metal],
    #     property_name="logK",
    # )
