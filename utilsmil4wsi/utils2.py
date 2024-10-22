from typing import Optional, Tuple
import torch
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils._subgraph import subgraph


# code borrowed from pytorch geometric framework

def dropout_node(edge_index: Tensor, p: float = 0.5,
                 num_nodes: Optional[int] = None,
                 training: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)
    Returns:
            edge_index (LongTensor): The edge indices.
            edge_mask (BoolTensor): The edge mask indicating which edges were retained.
            node_mask (BoolTensor): The node mask indicating which nodes were retained.
    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    # Check if dropout probability is valid
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')
    # Get the number of nodes from edge_index or num_nodes argument
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # If not in training mode or p=0, return all nodes and edges
    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    # Generate random probabilities for each node
    prob = torch.rand(num_nodes, device=edge_index.device)

    # Create a node mask based on the dropout probability
    node_mask = prob > p
    # Subgraph the original edge_index and get the corresponding edge mask
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    # Return the retained edge_index, edge_mask, and node_mask
    return edge_index, edge_mask, node_mask
