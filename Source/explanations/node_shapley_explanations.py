import copy
import itertools
import logging
import os

import networkx as nx
import numpy as np
import torch
from PIL import Image
from rdkit.Chem import rdDepictor
from torch_geometric.utils import to_networkx

from Source.explanations.rdkit_heatmaps import mapvalues2mol
from Source.explanations.rdkit_heatmaps.utils import transform2png
from Source.explanations.shapley import l_shapley
from Source.models.GCNN_FCNN.featurizers import Complex

general_logger = logging.getLogger("Explain")
general_logger.addHandler(logging.StreamHandler())
general_logger.setLevel(logging.INFO)


def create_path(path):
    if os.path.exists(path) or path == "":
        return
    head, tail = os.path.split(path)
    create_path(head)
    os.mkdir(path)


def classification_dataset(dataset, num_classes, thresholds=None):
    dataset = copy.deepcopy(dataset)
    all_values = np.array([graph.y["logK"].item() for graph in dataset])
    thresholds = thresholds or [np.percentile(all_values, (i + 1) * 100 / num_classes) for i in range(num_classes)]
    for graph in dataset:
        class_id = torch.tensor([num_classes - 1], dtype=torch.int64)
        for i, threshold in enumerate(thresholds[::-1]):
            if graph.y["logK"].item() < threshold:
                class_id = torch.tensor([num_classes - i - 1], dtype=torch.int64)
        graph.y = {"logK_class": class_id}
    return dataset, thresholds


def all_connected_coalitions(graph, min_nodes=1, max_nodes=None):
    G = to_networkx(graph, to_undirected=True)
    max_nodes = max_nodes or G.number_of_nodes()
    max_nodes = min(max_nodes, G.number_of_nodes())

    coalitions = []
    for nb_nodes in range(min_nodes, max_nodes + 1):
        for selected_nodes in itertools.combinations(G, nb_nodes):
            if nx.is_connected(G.subgraph(selected_nodes)):
                coalitions += [list(selected_nodes)]

    return coalitions


def visualize(mol, scores: list, save_path: str = None, normalize=False):
    scores = np.array(scores)
    rdDepictor.Compute2DCoords(mol)
    if normalize: scores = (scores - scores.mean()) / scores.std()

    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("atomNote", f"{scores[i]:.2f}")

    canvas = mapvalues2mol(mol, scores)
    img = transform2png(canvas.GetDrawingText())
    if save_path is not None: img.save(save_path)
    return img


def explain_by_shapley_nodes(model, complex: Complex) -> Image.Image:
    scores = [l_shapley(coalition=[node_id],
                        data=complex.graph,
                        local_radius=len(model.models[0].graph_sequential.conv_sequential),
                        value_func=lambda batch: model(batch.to(model.device))["logK"].detach().cpu(),
                        subgraph_building_method='zero_filling')
              for node_id in range(complex.mol.GetNumAtoms())]
    return visualize(mol=complex.mol, scores=scores)
