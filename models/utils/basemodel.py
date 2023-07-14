import torch
import torch.nn as nn
from utils.losses import *
from utils.utils2 import dropout_node


class Baseline(nn.Module):

    def __init__(self, args, state_dict_weights=None):
        """
        Baseline model for forward and loss computation.

        Args:
            args (Namespace): Arguments passed to the model.
            state_dict_weights (Dict, optional): State dictionary of weights. Defaults to None.
        """
        super().__init__()

        # Store the arguments
        self.args = args
        self.target = args.target
        self.lamb = args.lamb
        self.beta = args.beta
        self.dropout = args.dropout
        self.tau = args.temperature
        self.kl = args.kl
        self.residual = args.residual
        self.c_in = args.input_size
        self.classes = args.n_classes
        self.c_hidden = args.c_hidden
        self.add_bias = args.add_bias
        self.betapreg = args.preg
        self.max = args.max

        self.state_dict_weights = state_dict_weights

    def forward_scale(self, x: torch.Tensor, edge_index: torch.Tensor, gnnlayer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass at the scale level.

        Args:
            x (torch.Tensor): Input features.
            edge_index (torch.Tensor): Adjacency matrix.
            gnnlayer (torch.nn.Module): GNN module.

        Returns:
            x (torch.Tensor): Output features.
            edge_index (torch.Tensor): Adjacency matrix.
        """
        r = x
        if self.training and self.args.dropout:
            edge_index, _, _ = dropout_node(edge_index=edge_index)
        x = gnnlayer(x, edge_index)
        if self.residual:
            x = r+x
        else:
            x = x
        return x, edge_index

    def forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor, levels: torch.Tensor, childof: torch.Tensor, edge_index2: torch.Tensor = None, edge_index3: torch.Tensor = None):
        """
        Forward pass of the GNN model.

        Args:
            x (torch.Tensor): Input features.
            edge_index (torch.Tensor): List of vertex index pairs representing the edges in the graph.
            levels (torch.Tensor): Tensor indicating the levels of each node.
            childof (torch.Tensor): Tensor indicating the parent-child relationship between nodes.
            edge_index2 (torch.Tensor, optional): Additional edge index tensor. Defaults to None.
            edge_index3 (torch.Tensor, optional): Additional edge index tensor. Defaults to None.
        """
        NotImplementedError("forward_gnn error")

    def forward_mil(self, indecesperlevel: torch.Tensor, feats: torch.Tensor, results: dict):
        """
        Forward pass of the MIL model.

        Args:
            indecesperlevel (torch.Tensor): Tensor containing the indices per level.
            feats (torch.Tensor): Input features.
            results (dict): Dictionary to store the results.
        """
        NotImplementedError("forward_mil error")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, levels: torch.Tensor, childof: torch.Tensor, edge_index2: torch.Tensor = None, edge_index3: torch.Tensor = None):
        """forward model

        Args:
            x (torch.Tensor): input
            edge_index (torch.Tensor): adjecency matrix
            levels (torch.Tensor): scale level
            childof (torch.Tensor): interscale information
            edge_index2 (torch.Tensor, optional): adjecency matrix of single scale
            edge_index3 (torch.Tensor, optional): adjecency matrix of single scale

        Returns:
            results (dict): model output
        """
        feats, indecesperlevel, results = self.forward_gnn(
            x, edge_index, levels, childof, edge_index2, edge_index3)
        results = self.forward_mil(indecesperlevel, feats, results)
        return results

    def compute_loss(self, loss_module_instance: torch.nn.Module, results: dict, bag_label: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the model.

        Args:
            loss_module_instance (torch.nn.Module): Loss criterion.
            results (dict): Model output.
            bag_label (torch.Tensor): Ground truth labels.

        Returns:
            loss (torch.Tensor): Loss value.
        """
        loss = 0
        # Target level prediction
        y_instance_pred1, prediction_bag1, _, _ = results[self.target]
        higher_loss = CELoss(loss_module_instance,
                             y_instance_pred1, prediction_bag1, bag_label)
        loss += higher_loss
        # ------
        if self.kl is not None:
            # Second scale loss
            y_instance_pred2, prediction_bag2, _, _ = results[self.kl]
            lower_loss = CELoss(loss_module_instance, y_instance_pred2,
                                prediction_bag2, bag_label, lamb=(1.0-self.lamb))
            loss += lower_loss
            # -----
            # Distill patch level
            if self.max:
                mse_loss = computeMSE_max(
                    y_instance_pred2, y_instance_pred1, results["childof"])
            else:
                mse_loss = computeMSE(
                    y_instance_pred2, y_instance_pred1, results["childof"])
            loss += self.beta*mse_loss
            # ------
            kl_bag_loss = computeKL(
                prediction_bag2, prediction_bag1, self.tau, self.classes, self.add_bias)
            loss += self.lamb*kl_bag_loss
            # -------
        return loss

    def predict(self, results):
        """
        Generate predictions based on the model's output results.

        Args:
            results (dict): Model output results.

        Returns:
            higher_prediction (torch.Tensor): Higher-level predictions.
            lower_prediction (torch.Tensor): Lower-level predictions.
        """
        prediction_patch_higher, prediction_bag_higher, _, _ = results[self.target]

        if self.classes > 1:
            # For multi-class classification
            higher_prediction = torch.sigmoid(prediction_bag_higher)
            if self.kl is not None:
                # If second scale predictions are available
                _, prediction_bag_lower, A, B = results[self.kl]
                lower_prediction = 0.5 * \
                    torch.sigmoid(prediction_bag_lower)+0.5 * \
                    torch.sigmoid(prediction_bag_higher)[:, -1]
            else:
                lower_prediction = None
        else:
            # For binary classification
            higher_prediction = 0.5 * \
                torch.sigmoid(prediction_bag_higher)[
                    :, -1]+0.5 * torch.sigmoid(prediction_bag_higher)[:, -1]
            if self.kl is not None:
                # If second scale predictions are available
                _, prediction_bag_lower, _, _ = results[self.kl]
                lower_prediction = torch.sigmoid(prediction_bag_lower)[:, -1]
            else:
                lower_prediction = None
        return higher_prediction, lower_prediction
