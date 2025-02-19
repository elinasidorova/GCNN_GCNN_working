import warnings
from inspect import signature

import torch.optim.optimizer
from torch_geometric.data import Data

from Source.models.FCNN.model import FCNN
from Source.models.GCNN.model import GCNN
from Source.models.base_model import BaseModel
from Source.models.global_poolings import ConcatPooling


class GCNNGCNN(BaseModel):
    def __init__(self, metal_node_features, mol_node_features, targets,
                 mol_gcnn_params=None, metal_gcnn_params=None, global_pooling=ConcatPooling, post_fc_params=None,
                 use_out_sequential=False,
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super(GCNNGCNN, self).__init__(targets, use_out_sequential, optimizer, optimizer_parameters)
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        # preparing params for model blocks
        for p in [mol_gcnn_params, metal_gcnn_params, post_fc_params]:
            if "targets" in p and len(p["targets"]) > 0:
                warnings.warn(
                    "Not recommended to set 'targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["targets"] = ()

        mol_gcnn_params["input_features"] = mol_node_features
        metal_gcnn_params["node_features"] = metal_node_features
        self.mol_gcnn_sequential = GCNN(**mol_gcnn_params)
        self.metal_gcnn_sequential = GCNN(**metal_gcnn_params)

        self.global_pooling = global_pooling(input_dims=(self.mol_gcnn_sequential.output_dim,
                                                         self.metal_gcnn_sequential.output_dim))

        post_fc_params["input_features"] = self.global_pooling.output_dim

        self.post_fc_sequential = FCNN(**post_fc_params)

        self.last_common_dim = (self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1]
        self.configure_out_layer()

    def forward(self, graph):
        mol_graph = Data(x=graph.x_mol, edge_index=graph.edge_index_mol,
                         edge_attr=graph.edge_attr_mol, u=graph.u_mol,
                         batch=graph.batch)
        metal_graph = Data(x=graph.x_metal, edge_index=graph.edge_index_metal,
                           edge_attr=graph.edge_attr_metal, u=graph.u_metal,
                           batch=graph.batch)
        mol_x = self.mol_gcnn_sequential(mol_graph)
        metal_x = self.metal_gcnn_sequential(metal_graph)
        general = self.global_pooling(mol_x, metal_x)
        general = self.post_fc_sequential(general)
        if self.use_out_sequential:
            general = {target: sequential(general) for target, sequential in self.out_sequentials.items()}
        return general


class GCNN_GCNN(BaseModel):
    def __init__(self, molecule_node_features, solvent_node_features, targets,
                 solvent_gcnn_params=None, molecule_gcnn_params=None, global_pooling=ConcatPooling, post_fc_params=None,
                 use_out_sequential=True,
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super(GCNN_GCNN, self).__init__(targets, use_out_sequential, optimizer, optimizer_parameters)
        param_values = locals()
        self.config = {name: param_values[name] for name in signature(self.__init__).parameters.keys()}

        # preparing params for model blocks
        for p in [solvent_gcnn_params, molecule_gcnn_params, post_fc_params]:
            if "targets" in p and len(p["targets"]) > 0:
                warnings.warn(
                    "Not recommended to set 'targets' in model blocks as far as it doesn't affect anything",
                    DeprecationWarning)
            if "use_out_sequential" in p:
                warnings.warn("'use_out_sequential' parameter forcibly set to False", DeprecationWarning)
            p["use_out_sequential"] = False
            p["targets"] = ()

        solvent_gcnn_params["node_features"] = solvent_node_features
        molecule_gcnn_params["node_features"] = molecule_node_features
        self.solvent_gcnn_sequential = GCNN(**solvent_gcnn_params)
        self.molecule_gcnn_sequential = GCNN(**molecule_gcnn_params)

        # self.global_pooling = global_pooling(input_dims=(self.solvent_gcnn_sequential.output_dim,
        #                                                  self.molecule_gcnn_sequential.output_dim))
        self.global_pooling = global_pooling(input_dims=(self.solvent_gcnn_sequential.output_dim,
                                                         self.molecule_gcnn_sequential.output_dim))

        post_fc_params["input_features"] = self.global_pooling.output_dim

        self.post_fc_sequential = FCNN(**post_fc_params)

        self.last_common_dim = (self.global_pooling.output_dim, self.post_fc_sequential.output_dim)[-1]
        self.configure_out_layer()

    def forward(self, graph):
        solvent_graph = Data(x=graph.x_solvent, edge_index=graph.edge_index_solvent, batch=graph.batch_solvent)
        molecule_graph = Data(x=graph.x_molecule, edge_index=graph.edge_index_molecule, batch=graph.batch_molecule)
        solvent_x = self.solvent_gcnn_sequential(solvent_graph)
        molecule_x = self.molecule_gcnn_sequential(molecule_graph)
        general = self.global_pooling(solvent_x, molecule_x)
        general = self.post_fc_sequential(general)

        if self.use_out_sequential:
            general = {target: sequential(general) for target, sequential in self.out_sequentials.items()}
        return general
