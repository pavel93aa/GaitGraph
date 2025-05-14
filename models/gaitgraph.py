import torch
from torch_geometric.nn import GCNConv, global_mean_pool


class GaitGraph(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=3):
        """
        :param input_dim: number of features per node (e.g., x, y coordinates)
        :param hidden_dim: size of the hidden layer
        :param num_classes: number of psychological classes (Low, Normal, High)
        """
        super(GaitGraph, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if x.shape[0] == 0:
            print("Empty tensor x received")
            return torch.tensor([])

        # Check for NaNs or infinities in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected — replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize coordinates
        x -= x.mean(dim=0)
        x /= x.std(dim=0) + 1e-6

        try:
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
        except Exception as e:
            print(f"Error during forward pass: {e}")
            print(f"x.shape = {x.shape}, edge_index.max() = {edge_index.max()}, nodes_count = {x.shape[0]}")
            raise

        x = global_mean_pool(x, batch)
        return self.classifier(x)
