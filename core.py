import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from util import prepare_channel_direct_features, cp2array_risnet, compute_complete_channel, \
    compute_wmmse_v_v2, mmse_precoding
from torch.utils.data import Dataset
from joblib import Parallel, delayed, cpu_count
from util import discretize_phase


class RISnetPartialCSINOMA(nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.feature_dim = 10
        self.output_dim = 1
        self.info_dim = 32
        self.global_info_not_opposite_info_antennas = params["global_info_not_opposite_info_antennas"]
        self.global_info_not_opposite_info_users = params["global_info_not_opposite_info_users"]
        self.normalize_phase = params["normalize_phase"]
        self.ris_shape = params["ris_shape"]
        self.cluster_shape = 3

        if self.global_info_not_opposite_info_antennas:
            self.e_antennas123 = torch.ones((16, 16)).to(device) / 16
            self.e_antennas456 = torch.ones((144, 144)).to(device) / 144
            self.e_antennas7 = torch.ones((1296, 1296)).to(device) / 1296
        else:
            self.e_antennas123 = (torch.ones((16, 16)) - torch.eye(16)).to(device) / 15
            self.e_antennas456 = (torch.ones((144, 144)) - torch.eye(144)).to(device) / 143
            self.e_antennas7 = (torch.ones((1296, 1296)) - torch.eye(1296)).to(device) / 1295

        if self.global_info_not_opposite_info_users:
            self.e_users = torch.ones((params["num_users"], params["num_users"])).to(device) / params["num_users"]
        else:
            self.e_users = torch.ones((params["num_users"], params["num_users"])).to(device) - \
                           torch.eye(params["num_users"]).to(device) / (params["num_users"] - 1)

        self.l1 = nn.Conv1d(self.feature_dim, self.info_dim, 1).to(device)
        self.l2 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.l3 = nn.ModuleList([nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device) for i in range(9)])
        self.l4 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.l5 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.l6 = nn.ModuleList([nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device) for i in range(9)])
        self.l7 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)

        self.g1 = nn.Conv1d(self.feature_dim, self.info_dim, 1).to(device)
        self.g2 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.g3 = nn.ModuleList([nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device) for i in range(9)])
        self.g4 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.g5 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.g6 = nn.ModuleList([nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device) for i in range(9)])
        self.g7 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)

        self.layer8 = nn.Conv1d(self.info_dim * 2, 1, 1).to(device)

        self.resort_mat3 = resort([12, 12], [3, 3], 12, device)
        self.resort_mat6 = resort([36, 36], [3, 3], 108, device)

    def forward(self, channel):
        def process_layer(features, l, g, e_antennas):
            feature_l = F.relu(l(features))
            feature_g = F.relu(g(features)) @ e_antennas

            layer_output = torch.cat([feature_l, feature_g], dim=1)
            return layer_output

        def process_interpolation_layer(features, l, g, e_antennas, resort_mat):
            layer_output = torch.cat([process_layer(features, le, ge, e_antennas)
                                      for le, ge in zip(l, g)], dim=2)
            return layer_output @ resort_mat

        # Rotation invariance
        if self.normalize_phase:
            mean_phase = torch.mean(channel[:, 1::2, :, :], dim=[1, 2, 3], keepdim=True).detach()
            channel[:, 1::2, :, :] -= mean_phase

        f = process_layer(channel, self.l1, self.g1, self.e_antennas123)
        f = process_layer(f, self.l2, self.g2, self.e_antennas123)
        f = process_interpolation_layer(f, self.l3, self.g3, self.e_antennas123, self.resort_mat3)
        f = process_layer(f, self.l4, self.g4, self.e_antennas456)
        f = process_layer(f, self.l5, self.g5, self.e_antennas456)
        f = process_interpolation_layer(f, self.l6, self.g6, self.e_antennas456, self.resort_mat6)
        f = process_layer(f, self.l7, self.g7, self.e_antennas7)
        f = self.layer8(f) * np.pi

        return f


def resort(new_ris_shape, cluster_shape, old_num_cols, device="cpu"):
    # cluster_shape is the number of columns and rows in the **new, or expanded** cluster
    # old_num_cols
    num_cols = new_ris_shape[0] * new_ris_shape[1]
    resort_mat = torch.zeros((num_cols, num_cols))
    counter = 0
    for i in range(new_ris_shape[0]):
        for j in range(new_ris_shape[1]):
            filter_idx = ij2idx(i % cluster_shape[0], j % cluster_shape[1], cluster_shape[0])  # filter index in the cluster
            cluster_idx = ij2idx(np.floor(i / cluster_shape[0]),  # cluster index in the new ris
                                 np.floor(j / cluster_shape[1]),
                                 int(new_ris_shape[1] / cluster_shape[1]))
            old_idx = int(filter_idx * old_num_cols + cluster_idx)
            resort_mat[old_idx, counter] = 1  # (old index, new index)
            counter += 1
    return resort_mat.to(device)


def ij2idx(i, j, num_cols):
    return i * num_cols + j


class RISnetNOMA(nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.feature_dim = 10
        self.output_dim = 1
        self.info_dim = 32
        self.skip_connection = params["skip_connection"]
        self.global_info_not_opposite_info_antennas = params["global_info_not_opposite_info_antennas"]
        self.global_info_not_opposite_info_users = params["global_info_not_opposite_info_users"]
        self.normalize_phase = params["normalize_phase"]

        if self.global_info_not_opposite_info_antennas:
            self.e_antennas = (torch.ones((params["num_ris_antennas"], params["num_ris_antennas"])).to(device)
                               / params["num_ris_antennas"])
        else:
            self.e_antennas = (torch.ones((params["num_ris_antennas"], params["num_ris_antennas"])).to(device) -
                               torch.eye(params["num_ris_antennas"]).to(device) / (params["num_ris_antennas"] - 1))
        # self.e_antennas = self.e_antennas[None, None, :, :]

        if self.global_info_not_opposite_info_users:
            self.e_users = torch.ones((params["num_users"], params["num_users"])).to(device) / params["num_users"]
            self.norm_users = params["num_users"]
        else:
            self.e_users = (torch.ones((params["num_users"], params["num_users"])).to(device) -
                            torch.eye(params["num_users"]).to(device) / (params["num_users"] - 1))
        # self.e_users = self.e_users[None, None, :, :]

        self.l1 = nn.Conv1d(self.feature_dim, self.info_dim, 1).to(device)
        self.l2 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.l3 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.l4 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.l5 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.l6 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.l7 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)

        self.g1 = nn.Conv1d(self.feature_dim, self.info_dim, 1).to(device)
        self.g2 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.g3 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.g4 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.g5 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.g6 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)
        self.g7 = nn.Conv1d(self.info_dim * 2, self.info_dim, 1).to(device)

        self.layer8 = nn.Conv1d(self.info_dim * 2, 1, 1).to(device)

    def forward(self, channel):
        def process_layer(features, l, g):
            feature_l = F.relu(l(features))
            feature_g = F.relu(g(features)) @ self.e_antennas

            layer_output = torch.cat([feature_l, feature_g], dim=1)
            return layer_output

        f = process_layer(channel, self.l1, self.g1)
        if self.skip_connection:
            f1 = f
        f = process_layer(f, self.l2, self.g2)
        f = process_layer(f, self.l3, self.g3)
        if self.skip_connection:
            f3 = f
        f = process_layer(f, self.l4, self.g4)
        f = process_layer(f, self.l5, self.g5)
        if self.skip_connection:
            f = (f + f1) / 2
        f = process_layer(f, self.l6, self.g6)
        f = process_layer(f, self.l7, self.g7)
        if self.skip_connection:
            f = (f + f3) / 2
        f = self.layer8(f) * np.pi

        return f


class RTChannels(Dataset):
    def __init__(self, params, channel_tx_ris_pinv, device='cpu', test=False):
        self.params = params
        self.device = device
        if test:
            self.locations = torch.load(params["location_testing_path"], map_location=torch.device(device)).cfloat()
            self.group_definition = np.load(params["group_definition_testing_path"])
            self.channels_ris_rx = torch.load(params['channel_ris_rx_testing_path'],
                                              map_location=torch.device(device)).cfloat()

            self.channels_ris_rx = torch.reshape(self.channels_ris_rx, params['channel_ris_rx_original_shape_testing'])[:,
                                   : params['ris_shape'][0],
                                   : params['ris_shape'][1]]
            self.channels_ris_rx = torch.reshape(self.channels_ris_rx, (
                -1, 1, params['ris_shape'][0] * params['ris_shape'][1]))
            self.channels_direct = torch.load(params['channel_direct_testing_path'],
                                              map_location=torch.device(device)).cfloat()
        else:
            self.locations = torch.load(params["location_path"], map_location=torch.device(device)).cfloat()
            self.group_definition = np.load(params["group_definition_path"])
            self.channels_ris_rx = torch.load(params['channel_ris_rx_path'], map_location=torch.device(device)).cfloat()

            self.channels_ris_rx = torch.reshape(self.channels_ris_rx, params['channel_ris_rx_original_shape'])[:,
                                   : params['ris_shape'][0],
                                   : params['ris_shape'][1]]
            self.channels_ris_rx = torch.reshape(self.channels_ris_rx, (
                -1, 1, params['ris_shape'][0] * params['ris_shape'][1]))
            self.channels_direct = torch.load(params['channel_direct_path'], map_location=torch.device(device)).cfloat()

        self.channel_array = cp2array_risnet(self.channels_ris_rx,
                                             1 / params['std_ris'],
                                             params['mean_ris'],
                                             device=device)

        channels_direct_array = prepare_channel_direct_features(self.channels_direct, channel_tx_ris_pinv,
                                                                self.params, self.device)
        self.channel_array = torch.cat([self.channel_array, channels_direct_array], 1)
        self.test = test
        weights = np.random.random(self.group_definition.shape)
        self.weights = torch.tensor(weights / np.sum(weights, axis=1, keepdims=True), dtype=torch.float32).to(device)

    def __getitem__(self, item):
        user_indices = self.group_definition[item, :]
        weights = self.weights[item, :]
        weights_in_feature = weights[None, :, None].repeat((1, 1, self.params["num_ris_antennas"]))
        channel_features = torch.stack([self.channel_array[i, :, :] for i in user_indices], dim=1)
        channel_features = torch.cat([channel_features, weights_in_feature], dim=0)
        channels_ris_rx = torch.squeeze(self.channels_ris_rx[user_indices, :, :])
        locations = self.locations[user_indices, :]
        channels_direct = torch.squeeze(self.channels_direct[user_indices, :])
        return [item, channel_features, channels_ris_rx, channels_direct, locations, weights]

    def __len__(self):
        return self.group_definition.shape[0]


class RTChannelsNOMA(RTChannels):
    def __init__(self, params, channel_tx_ris_pinv, device='cpu', test=False):
        super().__init__(params, channel_tx_ris_pinv, device, test=test)
        self.data_rates = torch.rand(self.channel_array.shape[0]).to(device)

    def __getitem__(self, item):
        user_indices = self.group_definition[item, :]
        channel_features = torch.cat([self.channel_array[i, :, :] for i in user_indices], dim=0)
        data_rates = self.data_rates[user_indices]
        features = torch.cat([(data_rates - 0.5)[:, None].repeat([1, self.params["num_ris_antennas"]]),
                              channel_features], dim=0)
        channels_ris_rx = torch.squeeze(self.channels_ris_rx[user_indices, :, :])
        locations = self.locations[user_indices, :]
        channels_direct = torch.squeeze(self.channels_direct[user_indices, :])
        return [item, features, data_rates, channels_ris_rx, channels_direct, locations]
