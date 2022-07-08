import os.path as osp

import torch, json
import numpy as np 
import pandas as pd

from torch_geometric.data import Data, Dataset, DataLoader

class BatchNewDimData(Data):
     def __cat_dim__(self, key, item):
         if key == 'x':
             return None
         else:
             return super().__cat_dim__(key, item)


class IFTTTDataset(Dataset):
    def __init__(self, root, file_source= './ifttt_build_dataset/', transform=None, pre_transform=None):
        super(IFTTTDataset, self).__init__(root, transform, pre_transform)
        self.file_source = file_source

    @property
    def raw_file_names(self):
        return [self.file_source+'graph_edges.csv', self.file_source+'graph_properties.csv', self.file_source+'embedding.json']

    @property
    def processed_file_names(self):
        return [f'data_{graph_id}.pt' for graph_id in np.arange(8319)]

    def process(self):
        file_source= './ifttt_build_dataset/' # cannot use self.file_source
        edges = pd.read_csv(file_source+'graph_edges.csv')
        properties = pd.read_csv(file_source+'graph_properties.csv')
        with open(file_source+'embedding.json','r',encoding='utf8') as fp:
            embedding_dict = json.load(fp)
            embedding_dict = dict(sorted(embedding_dict.items(), key=lambda x: x[0]))

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = torch.tensor(label_dict[graph_id])

            edge_index = torch.tensor(np.array([src, dst]))
            x = torch.tensor([embedding_dict[str(i)] for i in embedding_dict.keys()])
            data = BatchNewDimData(x= x, edge_index = edge_index, y = label, num_nodes = 2339)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{graph_id}.pt'))
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data       


if __name__ == "__main__":
    IFTTTDataset =  IFTTTDataset('./ifttt_build_dataset/')
    # IFTTTDataset = [torch.load(osp.join('./ifttt_build_dataset/processed/', f'data_{0}.pt')) for idx in np.arange(8319)]
    print(IFTTTDataset.get(0))

    loader = DataLoader(IFTTTDataset, batch_size=32)
    batch = next(iter(loader))
    print(batch)


