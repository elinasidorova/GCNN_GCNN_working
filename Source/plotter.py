import plotly.graph_objs as go
import numpy as np
import os
import json
from itertools import chain

import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, \
    recall_score, matthews_corrcoef, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F

BINARY_CLASS_METRICS = {"confusion_matrix": (confusion_matrix, {}), "f1_score": (f1_score, {}),
                        "roc_auc_score": (roc_auc_score, {}), "precision_score": (precision_score, {}),
                        "recall_score": (recall_score, {}), "matthews_corrcoef": (matthews_corrcoef, {}),
                        "accuracy_score": (accuracy_score, {})}

MULTY_CLASS_METRICS = {"confusion_matrix": (confusion_matrix, {}), "f1_score": (f1_score, {"average": None}),
                       "matthews_corrcoef": (matthews_corrcoef, {}), "accuracy_score": (accuracy_score, {})}

REG_METRICS = {"r2_score": (r2_score, {}), "mean_squared_error": (mean_squared_error, {}),
               "mean_absolute_error": (mean_absolute_error, {})}


def calculate_metrics(train_true, train_pred, valid_true, valid_pred, test_true=None, test_pred=None,
                      mode="regression"):
    results = {}

    if mode == "binary_classification":
        train_pred = (torch.sigmoid(torch.tensor(train_pred.tolist())).numpy() > 0.5).astype(int)
        valid_pred = (torch.sigmoid(torch.tensor(valid_pred.tolist())).numpy() > 0.5).astype(int)
        test_pred = (torch.sigmoid(torch.tensor(test_pred.tolist())).numpy() > 0.5).astype(int)
        metrics = BINARY_CLASS_METRICS

    elif mode == "multy_classification":
        train_pred = F.softmax(torch.tensor(train_pred.tolist()), dim=-1).numpy().argmax(axis=-1)
        valid_pred = F.softmax(torch.tensor(valid_pred.tolist()), dim=-1).numpy().argmax(axis=-1)
        test_pred = F.softmax(torch.tensor(test_pred.tolist()), dim=-1).numpy().argmax(axis=-1)
        metrics = MULTY_CLASS_METRICS

    elif mode == "regression":
        metrics = REG_METRICS

    else:
        raise ValueError(
            "Invalid mode value, only 'regression', 'binary_classification' or 'multy_classification' are allowed")

    for metric_name in metrics:
        metric, parameters = metrics[metric_name]
        results["{}_{}".format("train", metric_name)] = metric(train_true, train_pred, **parameters)
        results["{}_{}".format("valid", metric_name)] = metric(valid_true, valid_pred, **parameters)
        if test_true is not None:
            results["{}_{}".format("test", metric_name)] = metric(test_true, test_pred, **parameters)

    for key in results.keys():
        if type(results[key]) is np.ndarray:
            results[key] = results[key].tolist()
        elif type(results[key]) is np.float32 or type(results[key]) is np.float64:
            results[key] = float(results[key])

    return results

# class Plotter:
#     def __init__(self):
#         pass
#
#     def make_cm(self, cm_data):
#         plt.figure(figsize=(10, 7))
#         sn.set(font_scale=1.4)
#         return sn.heatmap(cm_data, annot=True, annot_kws={"size": 16})
#
#     def make_regplot(self, true_vals, pred_vals):
#         plt.figure(figsize=(10, 7))
#         sn.set(font_scale=1.4)
#
#         x, y = pd.Series(np.asarray(true_vals).flatten(), name="{}_true".format(self.valuename)), \
#                pd.Series(np.asarray(pred_vals).flatten(), name="{}_pred".format(self.valuename))
#
#         return sn.regplot(x=x, y=y)
#
#     def make_plots_metrics(self, data):
#         results = {}
#         graphs = []
#
#         if self.mode == "classification":
#             for metric in CLASS_METRICS:
#                 results["{}_{}".format("train", metric)] = CLASS_METRICS[metric](data["train_true"],
#                                                                                  np.argmax(data["train_pred"],
#                                                                                            axis=-1))
#                 results["{}_{}".format("valid", metric)] = CLASS_METRICS[metric](data["valid_true"],
#                                                                                  np.argmax(data["valid_pred"],
#                                                                                            axis=-1))
#                 if self.test_set:
#                     results["{}_{}".format("test", metric)] = CLASS_METRICS[metric](data["test_true"],
#                                                                                     np.argmax(data["test_pred"],
#                                                                                               axis=-1))
#         elif self.mode == "regression":
#             for metric in REG_METRICS:
#                 results["{}_{}".format("train", metric)] = REG_METRICS[metric](data["train_true"], data["train_pred"])
#                 results["{}_{}".format("valid", metric)] = REG_METRICS[metric](data["valid_true"], data["valid_pred"])
#                 if self.test_set:
#                     results["{}_{}".format("test", metric)] = REG_METRICS[metric](data["test_true"], data["test_pred"])
#
#         for key in results.keys():
#             if "confusion" in key:
#                 results[key] = results[key].tolist()
#
#         if self.mode == "classification":
#             train_cm_fig = self.make_cm(pd.DataFrame(results["train_confusion_matrix"], range(2), range(2)))
#             valid_cm_fig = self.make_cm(pd.DataFrame(results["valid_confusion_matrix"], range(2), range(2)))
#             graphs.append(train_cm_fig)
#             graphs.append(valid_cm_fig)
#             if self.test_set:
#                 test_cm_fig = self.make_cm(pd.DataFrame(results["test_confusion_matrix"], range(2), range(2)))
#                 graphs.append(test_cm_fig)
#
#         elif self.mode == "regression":
#             true_regplot = self.make_regplot(data["train_true"], data["train_pred"])
#             valid_regplot = self.make_regplot(data["valid_true"], data["valid_pred"])
#             graphs.append(true_regplot)
#             graphs.append(valid_regplot)
#             if self.test_set:
#                 test_regplot = self.make_regplot(data["test_true"], data["test_pred"])
#                 graphs.append(test_regplot)
#
#         results = dict((k, float(v)) if isinstance(v, list) else (k, float(v)) for k, v in results.items())
#         return results, graphs
#
#     def make_mean_plots_metrics(self):
#         mean_res = {"train_true": [], "train_pred": [],
#                     "valid_true": [], "valid_pred": [],
#                     "test_true": [], "test_pred": []}
#
#         for fold in range(self.n_split):
#             with open(os.path.join(self.main_folder, "fold_{}".format(fold + 1), "data.json")) as jf:
#                 fold_res = json.load(jf)
#             if self.mode == "regression":
#                 mean_res["train_pred"].append(np.asarray(self.models[fold].predict(self.train_set)))
#                 mean_res["valid_pred"].append(fold_res["valid_pred"])
#                 if self.test_set:
#                     mean_res["test_pred"].append(fold_res["test_pred"])
#                 mean_res["valid_true"].append(fold_res["valid_true"])
#             elif self.mode == "classification":
#                 # mean_res["train_pred"].append(np.argmax(self.models[fold].predict(self.train_set), axis=-1).flatten())
#                 mean_res["train_pred"].append(self.models[fold].predict(self.train_set))
#                 mean_res["valid_pred"].append(np.asarray(fold_res["valid_pred"]).flatten())
#                 if self.test_set:
#                     mean_res["test_pred"].append(self.models[fold].predict(self.test_set))
#                 mean_res["valid_true"].append(fold_res["valid_true"])
#         mean_res["train_true"] = self.train_set.y
#         mean_res["valid_true"] = list(chain.from_iterable(mean_res["valid_true"]))
#         mean_res["train_pred"] = np.mean(mean_res["train_pred"], axis=0)
#         if self.test_set:
#             mean_res["test_true"] = self.test_set.y
#             mean_res["test_pred"] = np.mean(mean_res["test_pred"], axis=0)
#
#         if self.mode == "classification":
#             mean_res["valid_pred"] = list(chain.from_iterable(mean_res["valid_pred"]))
#             mean_res["valid_pred"] = np.asarray(mean_res["valid_pred"]).reshape(-1, 2)
#         elif self.mode == "regression":
#             mean_res["valid_pred"] = list(chain.from_iterable(mean_res["valid_pred"]))
#
#         mean_metrics, mean_graphs = self.make_plots_metrics(mean_res)
#         return mean_metrics, mean_graphs
#
#     def post_plots_metrics(self):
#         # TODO make check for training finishing here
#
#         for fold in range(self.n_split):
#             output_path = os.path.join(self.main_folder, "fold_{}".format(fold + 1))
#
#             with open(os.path.join(output_path, "data.json")) as jf:
#                 results, graphs = self.make_plots_metrics(json.load(jf))
#
#             with open("{}/metrics.json".format(output_path), "w") as fp:
#                 json.dump(results, fp)
#
#             for i in zip(graphs, ["Train", "Valid", "Test"]):
#                 i[0].figure.savefig(os.path.join(output_path, "{}_plot.png".format(i[1])))
#
#         mean_metrics, mean_graphs = self.make_mean_plots_metrics()
#         with open("{}/mean_metrics.json".format(self.main_folder), "w") as fp:
#             json.dump(mean_metrics, fp)
#
#         for i in zip(mean_graphs, ["Train", "Valid", "Test"]):
#             i[0].figure.savefig(os.path.join(self.main_folder, "{}_mean_plot.png".format(i[1])))
#
#
