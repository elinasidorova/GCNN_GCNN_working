import streamlit as st
from rdkit.Chem import Draw, MolFromSmiles
from streamlit_ketcher import st_ketcher

from Source.models.GCNN_FCNN.featurizers import Complex
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.trainer import ModelShell
from config import ROOT_DIR

MODEL = ModelShell(GCNN_FCNN, str(ROOT_DIR / "App_models" / "Am"))
AVAILABLE_METALS = ["Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",]

ketcher, conditions = st.columns(2)

with ketcher:
    smile_code = st_ketcher("")
    st.markdown(f"Smile code: ``{smile_code}``")

with conditions:
    metal = st.selectbox('Select metal', tuple(AVAILABLE_METALS))
    st.write('Selected metal:', metal)

    charge = st.text_input('Ion charge', 3)
    st.write('Selected charge:', charge)

    temperature = st.text_input('temperature', 20)
    st.write('Selected temperature:', temperature)

    ionic_str = st.text_input('Ionic strength', 0.1)
    st.write('Selected ionic strength:', ionic_str)

if st.button('Predict'):
    complex = Complex(mol=smile_code, metal=metal,
                      valence=int(charge), temperature=float(temperature), ionic_str=float(ionic_str))
    prediction = MODEL(complex.graph)["logK"].item()
    st.write("logK value: ", prediction)
    image = Draw.MolToImage(MolFromSmiles(smile_code))
    st.image(image, caption='Fragment importance')
