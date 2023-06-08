from janus import JANUS
from rdkit import RDLogger

from Source.models.GCNN_FCNN.featurizers import Complex
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.trainer import ModelShell
from config import ROOT_DIR

RDLogger.DisableLog("rdApp.*")
import selfies

METAL = "Cm"
VALENCE = 3
TEMPERATURE = 25
IONIC_STR = 0.1
MODEL = ModelShell(GCNN_FCNN, ROOT_DIR / "App_models/Cm")


def fitness_function(smi: str) -> float:
    """ User-defined function that takes in individual smiles
    and outputs a fitness value.
    """
    complex = Complex(mol=smi, metal=METAL, valence=VALENCE, temperature=TEMPERATURE, ionic_str=IONIC_STR)
    return MODEL(complex.graph)["logK"].item()


def custom_filter(smi: str):
    """ Function that takes in a smile and returns a boolean.
    True indicates the smiles PASSES the filter.
    """
    # smiles length filter
    if len(smi) > 81 or len(smi) == 0:
        return False
    else:
        return True


# all parameters to be set, below are defaults
params_dict = {
    # Number of iterations that JANUS runs for
    "generations": 200,

    # The number of molecules for which fitness calculations are done,
    # exploration and exploitation each have their own population
    "generation_size": 5000,

    # Number of molecules that are exchanged between the exploration and exploitation
    "num_exchanges": 5,

    # Callable filtering function (None defaults to no filtering)
    "custom_filter": custom_filter,

    # Fragments from starting population used to extend alphabet for mutations
    "use_fragments": True,

    # An option to use a classifier as selection bias
    "use_classifier": True,
}

# Set your SELFIES constraints (below used for manuscript)
default_constraints = selfies.get_semantic_constraints()
new_constraints = default_constraints
new_constraints['S'] = 2
new_constraints['P'] = 3
selfies.set_semantic_constraints(new_constraints)  # update constraints

# Create JANUS object.
agent = JANUS(
    work_dir='RESULTS',  # where the results are saved
    fitness_function=fitness_function,  # user-defined fitness for given smiles
    start_population="Data/sample_start_smiles.txt",  # file with starting smiles population
    **params_dict
)

# Alternatively, you can get hyperparameters from a yaml file
# Descriptions for all parameters are found in default_params.yml
# params_dict = utils.from_yaml(
#     work_dir = 'RESULTS',
#     fitness_function = fitness_function,
#     start_population = "./DATA/sample_start_smiles.txt",
#     yaml_file = 'default_params.yml',       # default yaml file with parameters
#     **params_dict                           # overwrite yaml parameters with dictionary
# )
# agent = JANUS(**params_dict)

# Run according to parameters
agent.run()  # RUN IT!
