import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/ywl/GNN-codes/BrainNetwork")
from Qua_2.model.cluster_pooling import DEC


class quadratic_perceptron(nn.Module):
    def __init__(self, hidden_in, hidden_out, activation, dropout=0.1):
        super().__init__()
        self.MLP_R = nn.Linear(hidden_in, hidden_out, bias=True)
        self.MLP_G = nn.Linear(hidden_in, hidden_out, bias=True)
        self.MLP_B = nn.Linear(hidden_in, hidden_out, bias=True)
        self.activation = {'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[activation]()
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature: torch.tensor, corr: torch.tensor):
        feature_R = self.MLP_R(feature)
        feature_G = self.MLP_G(feature)
        feature_B = torch.pow(feature, 2)
        feature_B = self.MLP_B(feature_B)
        x = feature_R * feature_G + feature_B + feature

        return self.activation(x)


class Quadratic_BN(nn.Module):
    def __init__(self, args, node_sz, time_series_sz, corr_pearson_sz, layers, dropout=0.,
                 cluster_num=4, pooling=True, orthogonal=True, freeze_center=True, project_assignment=True):
        super().__init__()
        forward_dim = corr_pearson_sz

        self.qp_layers = nn.ModuleList()
        for i in range(layers):
            qp_layer = quadratic_perceptron(hidden_in=corr_pearson_sz, hidden_out=corr_pearson_sz,
                                            activation=args.activation, dropout=dropout)
            self.qp_layers.append(qp_layer)

        self.activation = {'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[args.activation]()
        self.dropout = nn.Dropout(dropout)

        # orthogonal clustering readout
        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(forward_dim * node_sz, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, forward_dim * node_sz)
            )
            self.dec = DEC(cluster_number=cluster_num, hidden_dimension=forward_dim, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        if pooling:
            self.fc = nn.Sequential(
                nn.Linear(8 * cluster_num, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 2)
            )


    def forward(self,
                timeseries: torch.tensor,
                corr: torch.tensor):
        bz, node_sz, corr_sz = corr.shape

        topo = corr
        for qb_layer in self.qp_layers:
            topo = qb_layer(topo, corr)

        graph_level_topo, assignment = self.dec(topo)
        graph_level_topo = self.dim_reduction(graph_level_topo)
        graph_level_topo = graph_level_topo.reshape(bz, -1)
        result = self.fc(graph_level_topo)

        return result


