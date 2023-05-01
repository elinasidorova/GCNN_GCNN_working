import os
import gzip
import pickle

import pandas as pd
from pymatgen.core.structure import IStructure

# single dataframe for all structures (126335 rows, 2 cols)
# df["structure"] = [pymatgen.core.structure.IStructure,from_file(filename) for each filename]
# df["band_gap"] = ???

input_path = "Skipatom_data"
all_structures = []

for folder in os.listdir(input_path):
    for filename in os.listdir(os.path.join(input_path, folder)):
        print(folder, filename)
        path = os.path.join(input_path, folder, filename)
        all_structures += [IStructure.from_file(path)]

df = pd.DataFrame({"structure": all_structures})

output_path = r"Skipatom_data/AmBkCfCmRa_2022_24_04.pkl.gz"
with gzip.open(output_path, "wb") as file:
    pickle.dump(df, file)
