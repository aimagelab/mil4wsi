import torch
import submitit
import argparse
import joblib
import os
import glob
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from scipy.sparse import coo_matrix

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='prepare the dataset for pytorch geometric')
parser.add_argument('--source', type=str, help='origin folder')
parser.add_argument('--dest', type=str, help='destination folder')
parser.add_argument('--levels', type=int, nargs='+',
                    default=[1], help='destination folder')
{}
args = parser.parse_args()
dest = args.dest
source = args.source
levels = args.levels


def from_scipy_sparse_matrix(A):
    """
    Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.

    Returns:
        edge_index (torch.Tensor): Edge indices tensor of shape (2, num_edges).
        edge_weight (torch.Tensor): Edge attributes tensor of shape (num_edges,).
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0).to(torch.long)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


class MyOwnDataset(Dataset):
    """
    Custom dataset for processing and accessing data.

    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): A function/transform that takes in an
            object and returns a transformed version. Defaults to None.
        pre_transform (callable, optional): A function/transform that takes in an
            object and returns a transformed version. Defaults to None.
        type (str, optional): Type of the dataset. Defaults to "train".
    """

    def __init__(self, root, transform=None, pre_transform=None, type="train"):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.type = type
        self.lenght = len(glob.glob(os.path.join(
            self.processed_dir, self.type, "*data*.pt")))

    @property
    def raw_file_names(self):
        """
        Returns a list of raw file names in the dataset.

        Returns:
            list: List of raw file names.
        """
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        """
        Returns a list of processed file names in the dataset.

        Returns:
            list: List of processed file names.
        """
        return glob.glob(os.path.join(self.processed_dir, "*"))

    def len(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.lenght

    def process(self):
        """
        Process the dataset and save processed data.
        """
        bags = glob.glob(os.path.join(source, "*/*"))
        totLevels = levels
        for idx, bag in enumerate(bags):
            # Get data structure
            try:
                # Load patches and metadata
                patches = joblib.load(os.path.join(bag, "embeddings.joblib"))
                patch_level = patches["level"]
                patch_childof = patches["childof"]
                if "label" in patches.columns:
                    patch_label = patches["label"]
                else:
                    patch_label = [-1]
                patch_childof[patch_childof.isnull()] = -1
                embeddings = patches["embedding"]
                size = embeddings.shape[0]
                # Get X
                x = []
                for i in range(size):
                    x.append(torch.Tensor(np.matrix(embeddings[i])))
                X = torch.vstack(x)

                # Get full adjency matrix
                matrix_adj = torch.load(os.path.join(bag, "adj.th"))

                # Save main matrix
                matrix_edges = coo_matrix(matrix_adj)
                edge_index, edge_weight = from_scipy_sparse_matrix(
                    matrix_edges)

                # Prepare matrix without interscale connection
                matrix_adj_filtered = matrix_adj.clone()
                max = np.max(patch_level)
                min = np.min(patch_level)
                for level in range(max, min, -1):
                    matrix_adj_filtered[(patch_level.to_numpy() == level).nonzero()[
                        0], (patch_level.to_numpy() == level-1).nonzero()[0].reshape(-1, 1)] = 0
                    matrix_adj_filtered[(patch_level.to_numpy() == level-1).nonzero()[
                        0], (patch_level.to_numpy() == level).nonzero()[0].reshape(-1, 1)] = 0

                # Save matrix without interscale connection
                matrix_edges_filtered = coo_matrix(matrix_adj_filtered)
                edge_index_filtered, edge_weight = from_scipy_sparse_matrix(
                    matrix_edges_filtered)

                # Get sub-matrices
                if len(totLevels) > 1:
                    matrix_adj_2 = matrix_adj_filtered[patch_level ==
                                                       totLevels[0], :][:, patch_level == totLevels[0]]
                    matrix_adj_3 = matrix_adj_filtered[patch_level ==
                                                       totLevels[1], :][:, patch_level == totLevels[1]]
                    matrix_edges_2 = coo_matrix(matrix_adj_2)
                    edge_index_2, edge_weight = from_scipy_sparse_matrix(
                        matrix_edges_2)
                    matrix_edges_3 = coo_matrix(matrix_adj_3)
                    edge_index_3, edge_weight = from_scipy_sparse_matrix(
                        matrix_edges_3)
                else:
                    edge_index_2 = None
                    edge_index_3 = None

                # Save label
                label = os.path.basename(bag).split("_")[-1]
                if "0" in label or "1" in label:
                    label = int(label)
                else:
                    if label == "tumor":
                        label = 1
                    else:
                        label = 0
                # Create data object
                data = Data(x=X, edge_index=edge_index, edge_index_filtered=edge_index_filtered, edge_index_2=edge_index_2, edge_index_3=edge_index_3, childof=torch.LongTensor(patch_childof), level=torch.LongTensor(
                    patch_level), y=torch.LongTensor([label]), name=os.path.basename(bag), patch_label=torch.LongTensor(patch_label), x_coord=torch.LongTensor(patches["x"]), y_coord=torch.LongTensor(patches["y"]))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                os.makedirs(os.path.join(
                    self.processed_dir, "test"), exist_ok=True)
                os.makedirs(os.path.join(
                    self.processed_dir, "train"), exist_ok=True)
                if "test" in bag:
                    torch.save(data, os.path.join(
                        self.processed_dir, "test", 'data_{}.pt'.format(idx)))
                else:
                    torch.save(data, os.path.join(self.processed_dir,
                               "train", 'data_{}.pt'.format(idx)))
            except:
                print(bag)

    def get(self, idx):
        """
        Get the data object at the specified index.

        Args:
            idx (int): Index of the data object.

        Returns:
            Data: The data object at the specified index.
        """
        data = torch.load(os.path.join(
            self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


def prepareslide():
    """
    Prepare the slide dataset for processing.
    """
    global levels
    global dest
    global source
    dataset = MyOwnDataset(root=dest, type="train")
    print(dataset)


def main():
    """
    Execution setting.
    """
    log_folder = "log_test/%j"
    # Create an AutoExecutor instance with the log folder
    executor = submitit.AutoExecutor(folder=log_folder)
    # Update the parameters of the executor
    executor.update_parameters(
        slurm_partition="prod", name="data_prep", slurm_time=1200, cpus_per_task=5, mem_gb=20)
    # Submit the prepareslide function as a job
    jobs = executor.submit(prepareslide)


if __name__ == '__main__':
    main()
