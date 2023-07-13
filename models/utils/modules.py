import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch
import torch.nn.functional as F

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GATV2": geom_nn.GATv2Conv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}


class GNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GAT", dp_rate=0.1, heads=3):
        """Graph Neural Network (GNN) model.

        Args:
            c_in (int): Dimension of input features.
            c_hidden (int): Dimension of hidden features.
            c_out (int): Dimension of the output features, usually the number of classes in classification.
            num_layers (int): Number of "hidden" graph layers.
            layer_name (str): String specifying the type of graph layer to use.
            dp_rate (float): Dropout rate to apply throughout the network.
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT (Graph Attention Network))
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden

        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             heads=heads,
                             concat=False,
                             dropout=dp_rate)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Forward pass of the GNN model.

        Args:
            x (torch.Tensor): Input features per node.
            edge_index (torch.Tensor): List of vertex index pairs representing the edges in the graph (PyTorch geometric notation).

        Returns:
            torch.Tensor: Output features after passing through the GNN layers.
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True):  # K, L, N
        """
        Bag-level classifier class

        Args:
            input_size (int): Input feature size.
            output_class (int): Number of output classes.
            dropout_v (float): Dropout probability for the V layer.
            nonlinear (bool): Whether to use non-linear activations.
        """
        super(BClassifier, self).__init__()
        if nonlinear:
            # Non-linear transformation
            self.lin = nn.Sequential(
                nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            # Identity transformation
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        # 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class,
                             kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # Handle multiple classes without for loop
        # Sort class scores along the instance dimension, m_indices in shape N x C
        _, m_indices = torch.sort(c, 0, descending=True)
        # Select critical instances, m_feats in shape C x K
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        # Compute queries of critical instances, q_max in shape C x Q
        q_max = self.q(m_feats)
        # Compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = torch.mm(Q, q_max.transpose(0, 1))
        # Normalize attention scores, A in shape N x C,
        A = F.softmax(
            A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)
        # Compute bag representation, B in shape C x V
        B = torch.mm(A.transpose(0, 1), V)

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        """
        Fully connected layer module.

        Args:
            in_size (int): Input size.
            out_size (int): Output size. Defaults to 1.
        """
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        """
        Forward pass of the fully connected layer.

        Args:
            feats (torch.Tensor): Input features.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Tuple containing the input features and the output of the fully connected layer.
        """
        x = self.fc(feats)
        return feats, x


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        """
        MILNet module for multiple instance learning.

        Args:
            i_classifier (nn.Module): Instance-level classifier.
            b_classifier (nn.Module): Bag-level classifier.
        """
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        """
        Forward pass of the MILNet module.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple(torch.Tensor): Tuple containing the predicted classes, bag-level predictions,
                                 and intermediate variables A and B.
        """
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B


def init(model, state_dict_weights):
    """
    Initialize the model with the provided state_dict_weights.

    Args:
        model: The model to initialize.
        state_dict_weights: The state dictionary containing the model weights.

    Returns:
        Initialized model.
    """
    try:
        model.load_state_dict(state_dict_weights, strict=False)
    except:
        del state_dict_weights['b_classifier.v.1.weight']
        del state_dict_weights['b_classifier.v.1.bias']
        model.load_state_dict(state_dict_weights, strict=False)
    return model
