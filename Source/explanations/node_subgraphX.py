import copy
import itertools
import logging
import os
import sys

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.utils import to_networkx
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
from Source.explanations.rdkit_heatmaps import mapvalues2mol
from Source.explanations.rdkit_heatmaps.utils import transform2png
from Source.explanations.subgraphx import SubgraphX
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


test_metal = "Er"
device = torch.device("cuda")

path_to_sdf = str(ROOT_DIR / "Data/OneM_cond_adds/" / f"{test_metal}.sdf")
dataset = featurize_sdf_with_metal_and_conditions(path_to_sdf=path_to_sdf,
                                                  mol_featurizer=ConvMolFeaturizer(),
                                                  metal_featurizer=SkipatomFeaturizer(),
                                                  shuffle=False)
molecules = [(mol, p) for mol in Chem.SDMolSupplier(path_to_sdf) if mol is not None
             for p in mol.GetPropNames() if p.startswith("logK")]

ids = [10, 14, 16, 27, 37, 47, 55, 56, 61, 74, 75, 77, 104, 113, 118, 160, 162, 173, 181, 233, 242, 260, 264, 270, 286]
ids = list(sorted(ids, key=lambda i: dataset[i].x.shape[0]))

dataset = [dataset[i] for i in ids]
molecules = [molecules[i] for i in ids]

output_folder = str(ROOT_DIR / f"Output/comparison/")

create_path(os.path.join(output_folder, "node_shapley"))
create_path(os.path.join(output_folder, "node_subgraphX"))
create_path(os.path.join(output_folder, "сounterfactual"))

folder = [f for f in os.listdir(ROOT_DIR / "Output/WithCondAdd/5fold") if f.startswith(test_metal)][0]
train_folder = str(ROOT_DIR / "Output/WithCondAdd/5fold" / folder)

model = ModelShell(GCNN_FCNN, train_folder, device=device)
value_func = lambda batch: model(batch.to(device))["logK"].detach().cpu()

for graph_id, (graph, (molecule, prop_name)) in tqdm(enumerate(zip(dataset, molecules)), total=len(dataset),
                                                     desc="Molecules", leave=False):
    pred = model(graph.to(model.models[0].device))["logK"].item()
    true = graph.y["logK"].item()
    filename = f"smiles={Chem.MolToSmiles(molecule)}&prop_name={prop_name}&pred={pred}&true={true}.png"
    save_path = os.path.join(output_folder, "node_subgraphX", filename)
    if os.path.exists(save_path): continue

    explainer = SubgraphX(model, device=device,
                          value_func=value_func, reward_method='l_shapley',
                          min_atoms=max(1, molecule.GetNumAtoms() // 4),
                          verbose=True, )
    result = explainer.explain(graph)
    scores = {tuple(res["coalition"]): res["P"] for res in result}
    visualize(molecule, scores, save_path=save_path, normalize=True)
