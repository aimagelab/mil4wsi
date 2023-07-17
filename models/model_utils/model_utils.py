import torch.nn as nn


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        """
        Attention Network without Gating (2 fc layers)

        Args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to use dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net, self).__init__()
        self.module = [
            # Linear layer for attention
            nn.Linear(L, D),
            # Tanh activation function
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        # Sequential module
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        """
        Forward pass of the Attention Network

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: Output tensor and the input tensor itself
                -self.module(x) (torch.Tensor): N x n_classes
                -x (torch.Tensor): N x L
        """
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        """
        Attention Network with Sigmoid Gating (3 fc layers)

        Args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            # Linear layer for attention a
            nn.Linear(L, D),
            # Tanh activation function
            nn.Tanh()]

        self.attention_b = [
            # Linear layer for attention b
            nn.Linear(L, D),
            # Sigmoid activation function
            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        # Sequential module for attention a
        self.attention_a = nn.Sequential(*self.attention_a)
        # Sequential module for attention b
        self.attention_b = nn.Sequential(*self.attention_b)
        # Linear layer for attention c
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        """
        Perform the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor 

        Returns:
            tuple: Output tensor and the input tensor itself
                - A (torch.Tensor): N x n_classes
                - x (torch.Tensor): N x L
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    """
    Initialize Weights function.

    Args:
        module (torch.nn.Module): Module to initialize weights for using normal distribution
    """
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
