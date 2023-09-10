import torch
import glob
import os
from torch_geometric.data import Dataset
import torch_geometric.loader as geom_loader
from utilsmil4wsi.seed import seed_worker


def get_loaders(args):
    """
    Creates and returns data loaders for training, validation, and testing.

    Args:
        args (Namespace): Command-line arguments.

    Returns:
        tuple: A tuple containing the train, validation, and test data loaders.
    """
    # Create train and test datasets
    train_dataset = MyOwnDataset(root=args.datasetpath, type="train")
    test_dataset = MyOwnDataset(root=args.datasetpath, type="test")
    # Set the validation dataset as the test dataset
    val_dataset = test_dataset
    # Prepare dataloader
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
    """
    Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.

    Returns:
        tuple: A tuple containing the edge indices and edge attributes.
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0).to(torch.long)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, type="train"):
        """
        Custom dataset class for MyOwnDataset.

        Args:
            root (str): Root directory for the dataset.
            transform (callable, optional): A function/transform that takes in an
                torch_geometric.data.Data object and returns a transformed version.
                Defaults to None.
            pre_transform (callable, optional): A function/transform that takes in an
                torch_geometric.data.Data object and returns a transformed version.
                Defaults to None.
            type (str, optional): Type of the dataset (e.g., "train", "test").
                Defaults to "train".
        """
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.type = type
        self.bags = glob.glob(os.path.join(
            self.processed_dir, self.type, "*data*.pt"))
        self.data = [torch.load(bag) for bag in self.bags]
        self.length = len(self.bags)

    @property
    def processed_file_names(self):
        """
        Property to get the processed file names.

        Returns:
            List: List of processed file names.
        """
        return glob.glob(os.path.join(self.processed_dir, "*"))

    def len(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.length

    def get(self, idx):
        """
        Get the item at the given index.

        Args:
            idx (int): Index.

        Returns:
            torch_geometric.data.Data: Data object at the given index.
        """
        return self.data[idx]
