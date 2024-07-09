import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv, global_max_pool
from torch_scatter import scatter_sum
import copy


class Event_Critic_Net(torch.nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.up_conv1 = GATConv(state_size, hidden_size,
                                add_self_loops=False)
        # self.up_conv2 = GATConv(hidden_size, hidden_size)
        self.down_conv1 = GATConv(
            state_size, hidden_size, add_self_loops=False)
        # self.down_conv2 = GATConv(hidden_size, hidden_size)

        # self.up_conv1 = GCNConv(state_size, hidden_size)
        # self.up_conv2 = GCNConv(hidden_size, hidden_size)
        # self.down_conv1 = GCNConv(state_size, hidden_size)
        # self.down_conv2 = GCNConv(hidden_size, hidden_size)
        self.linear_mlp = torch.nn.Linear(hidden_size, 1)

    def forward(self, batch_up_data, batch_down_data):
        # for upstream event graph
        up_x, up_edge_index, up_batch = batch_up_data.x, batch_up_data.edge_index, batch_up_data.batch
        # message passing
        up_x = self.up_conv1(up_x.float(), up_edge_index)

        # 1. sum up the embeddings of all the nodes in each graph
        # embedding_up_x = global_mean_pool(up_x, up_batch)

        # 2. only keep the embedding of the self node
        # get the number of nodes in each graph in the batch
        up_num_nodes_per_graph = scatter_sum(up_batch.new_ones(
            batch_up_data.num_nodes), up_batch, dim=0)
        # compute the indices of the self node in each graph
        up_self_index = torch.cumsum(up_num_nodes_per_graph, dim=0) - 1
        # get the embedding of the self node in each graph
        up_self_node_embed = up_x[up_self_index]
        # embedding_up_x = F.sigmoid(up_self_node_embed)
        embedding_up_x = torch.sigmoid(up_self_node_embed)
        # for downstream event graph
        # same as above
        down_x, down_edge_index, down_batch = batch_down_data.x, batch_down_data.edge_index, batch_down_data.batch
        down_x = self.down_conv1(down_x.float(), down_edge_index)

        # 1. sum up the embeddings of all the nodes in each graph
        # embedding_down_x = global_mean_pool(down_x, down_batch)

        # 2. only keep the embedding of the self node
        down_num_nodes_per_graph = scatter_sum(down_batch.new_ones(
            batch_down_data.num_nodes), down_batch, dim=0)
        down_self_index = torch.cumsum(down_num_nodes_per_graph, dim=0) - 1
        down_self_node_embed = down_x[down_self_index]
        embedding_down_x = torch.sigmoid(down_self_node_embed)

        x = embedding_up_x + embedding_down_x
        # x = torch.concatenate(
        #     [embedding_up_x, emedding_down_x], dim=1)
        x = self.linear_mlp(x)

        return x
