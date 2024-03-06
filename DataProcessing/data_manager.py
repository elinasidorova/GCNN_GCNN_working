import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class DataManager(object):
    def __init__(self, file: str) -> None:
        self.counter = 0
        self.file = file
        self.cols = ['name', 'logK', 'T', 'solvent', 'I', 'method', 'doi', 'smiles', 'comment']
        try:
            self.data = pd.read_csv(file)
        except FileNotFoundError:
            self.data = pd.DataFrame(columns=self.cols)

    def refresh_counter(self):
        self.counter += 1

    def get_counter(self):
        return self.counter

    def get_cols(self):
        return self.cols

    def get_data(self) -> pd.DataFrame:
        return self.data

    @staticmethod
    def from_keyboard():
        new_data = {
            'name': [input("name: ")],
            'logK': [input("logK: ")],
            'T': [input("T, K: ") or 298.0],
            'solvent': [input("solvent: ") or "water"],
            'I': [input("I: ") or 0.0],
            'method': [input("method [spec, pm, comp(basis, func)]: ")],
            'doi': [input("DOI: ")],
            'smiles': [input("SMILES: ") or "?"],
            'comment': [input("short comment (if needed): ")]}
        return new_data

    def add_data(self, new_data: dict) -> None:
        new_df = pd.DataFrame(new_data, index=[0])
        self.data = pd.concat([self.data, new_df], ignore_index=True)
        self.refresh_counter()

    def get_amount(self):
        return self.data.index[-1] + 1

    def save_data(self) -> None:
        self.data.to_csv(self.file, index=False)

    def get_molecules(self) -> list:
        return self.data.name.to_list()


class Inspection(DataManager):
    def __init__(self):
        super().__init__()

    def _find_with(self, column: str, value) -> pd.Series or None:
        result = self.data[self.data[column] == value]
        return result

    def find_with_name(self, name):
        return self._find_with('name', name)

    def find_with_doi(self, doi):
        return self._find_with('doi', doi)

    def find_with_index(self, index):
        return self.data.iloc[index]
