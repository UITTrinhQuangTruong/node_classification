from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T


def load_data(name, transform):
    if transform:
        transform_func = T.ToSparseTensor()
    else:
        transform_func = None
    if name == 'cora':
        return Planetoid("Planetoid", name="Cora", transform=transform_func)
    else:
        return PygNodePropPredDataset(name=name,
                                      transform=transform_func)
