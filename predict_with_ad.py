import sys
from rdkit import Chem
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch_geometric.data import Batch
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

sys.path.append("./Source")

from Source.knn_ad import get_sdfs_ad
from Source.Predictor import Predictor
from Source.data import train_test_valid_split
from Source.featurizers.featurizers import featurize_sdf

path_to_sdf = "/home/cairne/PythonProj/SmartChemDesign/mol_torch_model/Data/An_converted/Am_ML.sdf"
path_to_model = "/home/cairne/PythonProj/SmartChemDesign/mol_torch_model/Output/Results_Am_logK_regression_2022_07_19_19_19_07"

valuenames = ["logK"]
output_path = "Output"
output_mark = "Am_logK_pure"
n_split = 5

mols = Chem.SDMolSupplier(path_to_sdf)
featurized = featurize_sdf(path_to_sdf, valuenames)
folds, test_loader = train_test_valid_split(featurized, n_split, test_ratio=0.2, batch_size=5, subsample_size=False,
                                            return_test=True)
dataset_size = len(mols)
ids = range(dataset_size)
train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=14)

predictor = Predictor(model_folder=path_to_model)
predicted = pd.DataFrame.from_dict(predictor.predict(test_loader))
predicted['average'] = predicted.mean(numeric_only=True, axis=1)
test_true = Batch.from_data_list(test_loader.dataset).y

ad = get_sdfs_ad(f"/home/cairne/PythonProj/SmartChemDesign/mol_torch_model/Data/An_converted/Am_ML_test_new.sdf",
                 f"/home/cairne/PythonProj/SmartChemDesign/mol_torch_model/Data/An_converted/Am_ML_train_new.sdf")
print(ad)
y = predicted['average']
x = test_true

mse = math.sqrt(mean_squared_error(y, x))
mae = mean_absolute_error(y, x)

import plotly.graph_objects as go

in_ad_x, in_ad_y = [val for i, val in enumerate(x) if ad[i]], [val for i, val in enumerate(y) if ad[i]]
out_ad_x, out_ad_y = [val for i, val in enumerate(x) if not ad[i]], [val for i, val in enumerate(y) if not ad[i]]

fig = go.Figure()
fig.add_trace(go.Scatter(x=in_ad_x, y=in_ad_y,
                         mode='markers',
                         name='Inside AD'))
fig.add_trace(go.Scatter(x=out_ad_x, y=out_ad_y,
                         mode='markers',
                         name='Outside AD'))

fig.add_annotation(text=f"$R^2 = {r2_score(x, y):.2f}$",
                   xref="paper", yref="paper",
                   x=0.2, y=0.8, showarrow=False)

regr = linear_model.LinearRegression()
X = np.array(x).reshape(-1, 1)
Y = np.array(y).reshape(-1, 1)
regr.fit(X, Y)
coef = regr.coef_
inter = regr.intercept_
Y_pred = regr.predict(X).reshape(1, -1)
fig.add_trace(go.Scatter(x=X.reshape(1, -1)[0], y=Y_pred[0],
                         mode='lines',
                         name='Fit',
                         line=dict(width=2, color='rgb(0, 0, 0)'),
                         showlegend=False))

regr_1 = linear_model.LinearRegression()
regr_1.coef_ = coef
regr_1.intercept_ = inter + coef * mae
Y_pred_1 = regr_1.predict(X).reshape(1, -1)
fig.add_trace(go.Scatter(x=X.reshape(1, -1)[0], y=Y_pred_1[0],
                         mode='lines',
                         name='Fit',
                         line=dict(shape='linear', color='rgb(0, 0, 0)', dash='dash', width=0.5),
                         showlegend=False))

regr_2 = linear_model.LinearRegression()
regr_2.coef_ = coef
regr_2.intercept_ = inter - coef * mae
Y_pred_2 = regr_2.predict(X).reshape(1, -1)
fig.add_trace(go.Scatter(x=X.reshape(1, -1)[0], y=Y_pred_2[0],
                         mode='lines',
                         name='Fit',
                         line=dict(shape='linear', color='rgb(0, 0, 0)', dash='dash', width=0.5),
                         showlegend=False))

fig.update_layout({"plot_bgcolor": 'rgba(0,0,0,0)'})
fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
fig.update_xaxes(showgrid=True, gridwidth=0.3, gridcolor='gray')
fig.update_yaxes(showgrid=True, gridwidth=0.3, gridcolor='gray')
fig.update_layout(
    title="Am model transfer quality",
    xaxis_title="logK true",
    yaxis_title="logK predicted", font=dict(size=15))
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="top",
    y=1.13,
    xanchor="right",
    x=1
))
fig.update_layout(title_x=0.5)
fig.write_image(f"Am_test_with_ad_tr.png")
