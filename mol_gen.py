import os
from datetime import datetime

from janus import JANUS
from rdkit import RDLogger
from rdkit import Chem
from rdkit import DataStructs

from Source.models.GCNN_FCNN.featurizers import Complex
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.trainer import ModelShell
from config import ROOT_DIR

RDLogger.DisableLog("rdApp.*")

METAL_1 = "Am"
METAL_2 = "Cm"
TARGET_MOLECULE = "C12C=CC(C(=O)N(CC)C3C=CC(CC)=CC=3)=NC1=C1N=C(C(N(C3C=CC(CC)=CC=3)CC)=O)C=CC1=CC=2"
TARGET_MOLECULE_FP = Chem.RDKFingerprint(Chem.MolFromSmiles(TARGET_MOLECULE))
SIMILARITY_THRESHOLD = 0.8
VALENCE = 3
TEMPERATURE = 25
IONIC_STR = 0.1

MODEL = ModelShell(GCNN_FCNN, ROOT_DIR / "App_models/Y_Sc_f-elements_5fold_regression_2023_06_10_07_58_17")
STARTING_POPULATION_PATH = "Data/merged_start_smiles.txt"
MAIN_OUT_DIR = "Generation_results"


def custom_filter(smi: str):
    """ Function that takes in a smile and returns a boolean.
    True indicates the smiles PASSES the filter.
    """
    fp_arr = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
    if len(smi) > 120 or len(smi) < 5:
        return False
    elif "+" in smi or "-" in smi:
        return False
    elif DataStructs.TanimotoSimilarity(TARGET_MOLECULE_FP, fp_arr) < SIMILARITY_THRESHOLD:
        return False
    else:
        return True


params_dict = {
    # Number of iterations that JANUS runs for
    "generations": 100,

    # The number of molecules for which fitness calculations are done,
    # exploration and exploitation each have their own population
    "generation_size": 5000,

    # Number of molecules that are exchanged between the exploration and exploitation
    "num_exchanges": 5,

    # Callable filtering function (None defaults to no filtering)
    "custom_filter": custom_filter,

    # Fragments from starting population used to extend alphabet for mutations
    "use_fragments": False,  # Causes errors in selfies encoding

    # An option to use a classifier as selection bias
    "use_classifier": True,
}


def fitness_function(smi: str) -> float:
    """ User-defined function that takes in individual smiles
    and outputs a fitness value.
    """
    complex_1 = Complex(mol=smi, metal=METAL_1, valence=VALENCE, temperature=TEMPERATURE, ionic_str=IONIC_STR)
    complex_2 = Complex(mol=smi, metal=METAL_2, valence=VALENCE, temperature=TEMPERATURE, ionic_str=IONIC_STR)

    return MODEL(complex_1.graph)["logK"].item() / MODEL(complex_2.graph)["logK"].item()


time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
WORKING_DIR = os.path.join(MAIN_OUT_DIR, f"{METAL_1}_{METAL_2}_{TEMPERATURE}_"
                                         f"{VALENCE}_{IONIC_STR}_{SIMILARITY_THRESHOLD}_{time_mark}")
os.mkdir(WORKING_DIR)
with open(os.path.join(WORKING_DIR, "target_mol"), "w") as f:
    f.write(TARGET_MOLECULE)

# Create JANUS object.
agent = JANUS(
    work_dir=WORKING_DIR,  # where the results are saved
    fitness_function=fitness_function,  # user-defined fitness for given smiles
    start_population=STARTING_POPULATION_PATH,  # file with starting smiles population
    explr_num_mutations=20,
    **params_dict
)

agent.run()
