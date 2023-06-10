import streamlit as st
from streamlit_ketcher import st_ketcher

from Source.explanations.exmol_explanations import explain_by_exmol
from Source.explanations.node_shapley_explanations import explain_by_shapley_nodes
from Source.models.GCNN_FCNN.featurizers import Complex
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.trainer import ModelShell
from config import ROOT_DIR

MODEL = ModelShell(GCNN_FCNN, str(ROOT_DIR / "App_models" / "Y_Sc_f-elements_5fold_regression_2023_06_10_07_58_17"))
AVAILABLE_METALS = ["Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", ]

ketcher, conditions = st.columns(2)

with ketcher:
    smile_code = st_ketcher("")
    st.markdown(f"Smile code: ``{smile_code}``")

with conditions:
    metal = st.selectbox('Select metal', tuple(AVAILABLE_METALS))
    charge = st.text_input('Ion charge', 3)
    temperature = st.text_input('Temperature', 20)
    ionic_str = st.text_input('Ionic strength', 0.1)

if st.button('Predict'):
    complex = Complex(mol=smile_code, metal=metal,
                      valence=int(charge), temperature=float(temperature), ionic_str=float(ionic_str))
    prediction = MODEL(complex.graph)["logK"].item()
    st.latex("\log_{10} K = %lf" % prediction, help="Estimated stability constant value")

    with st.expander('Fragment importance'):
        with st.spinner('Investigating fragment importance...'):
            image = explain_by_exmol(MODEL, complex)
        st.image(image)
        """
        First, we create a set of similar molecules (i. e. a chemical space around the given molecule)
        by mutating it's SELFIES representation.
        
        Then, for every molecule in set we calculate _logK_ using our model.
        Also for every molecule we generate EGFP features (binary vector that encode instance-based substructures) 
        and trying to approximate depandance of _logK_ from ECFP features by linear function.
        This gives us a vector of coefficients, each of them corresponds to the importance 
        of particular substructure for predicted _logK_.
        
        Thus, we can visualize these "importance values" as a barplot.
        
        For more information: https://chemrxiv.org/engage/chemrxiv/article-details/6273f42c59f0d6ba6984f315
        """

    with st.expander('Atom contributions'):
        with st.spinner('Calculating atom contributions...'):
            image = explain_by_shapley_nodes(MODEL, complex)
        st.image(image)
        """
        The Shapley value is the marginal contribution of each node to the overall prediction.
        A positive value means that the node increases the final prediction, a negative value means that it decreases.
        For more information: https://en.wikipedia.org/wiki/Shapley_value
        """
