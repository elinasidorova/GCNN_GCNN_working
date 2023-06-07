import copy
import itertools
import logging
import os

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from Source.explanations.rdkit_heatmaps import mapvalues2mol
from Source.explanations.rdkit_heatmaps.utils import transform2png
from Source.explanations.shapley import l_shapley
from Source.models.GCNN_FCNN.featurizers import featurize_sdf_with_metal_and_conditions, SkipatomFeaturizer
from Source.models.GCNN_FCNN.model import GCNN_FCNN
from Source.models.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.trainer import ModelShell
from config import ROOT_DIR

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


def visualize(mol, scores, save_path=None, normalize=False):
    all_scores = [[] for _ in range(mol.GetNumAtoms())]
    for coalition, score in scores.items():
        for node_id in coalition:
            all_scores[node_id] += [score]
    scores = np.array([np.mean(node_scores) for node_scores in all_scores])
    if normalize: scores = (scores - scores.mean()) / scores.std()

    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("atomNote", f"{scores[i]:.2f}")

    canvas = mapvalues2mol(mol, scores)
    img = transform2png(canvas.GetDrawingText())
    if save_path is not None: img.save(save_path)
    return img

all_metals = [f.split("_")[0] for f in os.listdir(ROOT_DIR / "Output/WithCondAdd/5fold")]

for test_metal in tqdm(all_metals, desc="Metals"):
    device = torch.device("cuda")

    path_to_sdf = str(ROOT_DIR / "Data/OneM_cond_adds/" / f"{test_metal}.sdf")
    output_folder = str(ROOT_DIR / f"Output/full_subgraph/{test_metal}")
    create_path(output_folder)
    folder = [f for f in os.listdir(ROOT_DIR / "Output/WithCondAdd/5fold") if f.startswith(test_metal)][0]
    train_folder = str(ROOT_DIR / "Output/WithCondAdd/5fold" / folder)

    model = ModelShell(GCNN_FCNN, train_folder, device=device)
    value_func = lambda batch: model(batch.to(device))["logK"].detach().cpu()
    dataset = featurize_sdf_with_metal_and_conditions(path_to_sdf=path_to_sdf,
                                                      mol_featurizer=ConvMolFeaturizer(),
                                                      metal_featurizer=SkipatomFeaturizer(),
                                                      shuffle=False)
    molecules = [(mol, p) for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None
                 for p in mol.GetPropNames() if p.startswith("logK")]

    for graph_id, (graph, (molecule, prop_name)) in enumerate(zip(dataset, molecules)):
        coalitions = all_connected_coalitions(graph, min_nodes=1, max_nodes=None)
        scores = {tuple(coalition): l_shapley(coalition=coalition, data=graph,
                                              local_radius=len(model.models[0].graph_sequential.conv_sequential),
                                              value_func=value_func,
                                              subgraph_building_method='zero_filling')
                  for coalition in tqdm(coalitions, desc="Scoring", leave=False)}
        visualize(molecule, scores, save_path=os.path.join(output_folder, f"{graph_id}. {prop_name}.png"))
        break
    break
