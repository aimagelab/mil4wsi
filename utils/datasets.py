import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import glob
import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import torch_geometric.loader as geom_loader
from utils.seed import seed_worker




def calculate_indeces(dataset):
    trainloader = geom_loader.DataLoader(dataset, batch_size=1, shuffle=True,worker_init_fn=seed_worker)
    indeces=[]
    labels=[]
    for idx,data in enumerate(trainloader):
        label,index=data[0].y,data[1]
        indeces.append(index.numpy()[0])
        labels.append(label.numpy()[0])
    return pd.DataFrame({"0":indeces,"1":labels})




def get_loaders(args):
    train_dataset= MyOwnDataset(root="data/"+args.dataset+"Graph_"+args.scale,type="train")
    if args.dataset=="cam":
        indeces=pd.read_csv("csvs/indeces.csv")
    else:
        indeces=pd.read_csv("csvs/lung_indices.csv")
    from sklearn.model_selection import train_test_split
    indeces_train, indeces_val, _, _ = train_test_split(indeces["0"].to_numpy(), indeces["1"],
                                                    stratify=indeces["1"],
                                                    test_size=0.3)
    val_dataset=torch.utils.data.Subset(train_dataset, indeces_val)
    train_dataset=torch.utils.data.Subset(train_dataset, indeces_train)
    test_dataset= MyOwnDataset(root="data/"+args.dataset+"Graph_"+args.scale,type="test")
    g = torch.Generator()
    g.manual_seed(args.seed)

    graph_val_loader = geom_loader.DataLoader(val_dataset, batch_size=1, shuffle=True,generator=g,worker_init_fn=seed_worker)
    graph_test_loader = geom_loader.DataLoader(test_dataset, batch_size=1,generator=g,worker_init_fn=seed_worker)
    graph_train_loader = geom_loader.DataLoader(train_dataset, batch_size=1, shuffle=True,generator=g,worker_init_fn=seed_worker)
    return graph_train_loader,graph_val_loader,graph_test_loader

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
        self.type=type
        self.bags=glob.glob(os.path.join(self.processed_dir,self.type, "*data*.pt"))
        self.lenght = len(self.bags)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return glob.glob(os.path.join(self.processed_dir, "*"))

    def len(self):
        return self.lenght

    def process(self):
        bags=glob.glob("data/feats/camdinoLevels3/*")
        for idx,bag in enumerate(bags):
            patches= pd.read_csv(os.path.join(bag,"embeddings.csv"))#["embedding"]
            patch_level=patches["level"]
            embeddings= patches["embedding"]#[patch_level.isin(levels)].reset_index(drop=True)
            len= embeddings.shape[0]
            x=[]
            for i in range(len):
                x.append(torch.Tensor(np.matrix(embeddings[i])))
            X=torch.vstack(x)
            matrix_adj= torch.load(os.path.join(bag,"adj.th"))
            #matrix_adj=matrix_adj[patch_level.isin(levels)][:,patch_level.isin(levels)]
            matrix_edges = coo_matrix(matrix_adj)
            edge_index, edge_weight = from_scipy_sparse_matrix(matrix_edges)
            label= os.path.basename(bag).split("_")[-1]
            if label=="tumor":
                label=1
            else:
                label=0
            # Read data from `raw_path`.
            data = Data(x=X,edge_index=edge_index, y=torch.LongTensor([label]))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            os.makedirs(os.path.join(self.processed_dir,"test"),exist_ok=True)
            os.makedirs(os.path.join(self.processed_dir,"train"),exist_ok=True)
            if "test" in bag:
                torch.save(data, os.path.join(self.processed_dir,"test", 'data_{}.pt'.format(idx)))
            else:
                torch.save(data, os.path.join(self.processed_dir,"train", 'data_{}.pt'.format(idx)))
        self.lenght = len(glob.glob(os.path.join(self.processed_dir, "*","*")))

    def get(self, idx):
        data = torch.load(self.bags[idx])
        return data
