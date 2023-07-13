import torch
import glob
import os
from torch_geometric.data import Dataset
import torch_geometric.loader as geom_loader
from utils.seed import seed_worker


def get_loaders(args):
    train_dataset = MyOwnDataset(root=args.datasetpath, type="train")
    test_dataset = MyOwnDataset(root=args.datasetpath, type="test")
    val_dataset = test_dataset
    # prepapare dataloader
    g = torch.Generator()
    g.manual_seed(args.seed)
    graph_val_loader = geom_loader.DataLoader(
        val_dataset, batch_size=1, shuffle=True, generator=g, worker_init_fn=seed_worker)
    graph_test_loader = geom_loader.DataLoader(
        test_dataset, batch_size=1, generator=g, worker_init_fn=seed_worker)
    graph_train_loader = geom_loader.DataLoader(
        train_dataset, batch_size=1, shuffle=True, generator=g, worker_init_fn=seed_worker)
    return graph_train_loader, graph_val_loader, graph_test_loader


def from_scipy_sparse_matrix(A):
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0).to(torch.long)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, type="train"):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.type = type
        self.bags = glob.glob(os.path.join(
            self.processed_dir, self.type, "*data*.pt"))
        self.data = [torch.load(bag) for bag in self.bags]
        self.lenght = len(self.bags)

    @property
    def processed_file_names(self):
        return glob.glob(os.path.join(self.processed_dir, "*"))

    def len(self):
        return self.lenght

    def get(self, idx):
        return self.data[idx]
