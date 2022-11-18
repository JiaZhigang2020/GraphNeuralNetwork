import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset


class Node:
    def __init__(self, name, node_index, node_type):
        self.name = name
        self.node_index = node_index
        self.node_type = node_type
        self.interaction_list = []
        self.embedded_vector = []


class LncRNA(Node):
    def __init__(self, name, node_index, node_type):
        Node.__init__(self, name, node_index, node_type)


class Protein(Node):
    def __init__(self, name, node_index, node_type):
        Node.__init__(self, name, node_index, node_type)


class LncRNAProteinInteraction:
    def __init__(self, lncRNA, protein, y:int, interaction_nodes_index_set):
        self.lncRNA = lncRNA
        self.protein = protein
        self.y = y
        self.key = interaction_nodes_index_set


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, interaction_list=None, set_interactionKey_can_use=None, set_interactionKey_cannot_use=None, transform=None, pre_transform=None):
        self.interaction_list = interaction_list
        self.set_interactionKey_can_use = set_interactionKey_can_use
        self.set_interactionKey_cannot_use = set_interactionKey_cannot_use
        self.sum_nodes = 0.0
        super(MyOwnDataset, self).__init__(root, transform, pre_transform, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     pass

    def process(self):
        # Read data into huge `Data` list.
        if self.set_interactionKey_can_use != None:
            num_data = len(self.set_interactionKey_can_use)
            data_list = []
            count = 0
            for interaction in self.interaction_list:
                interaction_nodes_list = (interaction.lncRNA.node_index, interaction.protein.node_index)
                if interaction_nodes_list in self.set_interactionKey_can_use:
                    data = self.get_local_subgraph(interaction)
                    data_list.append(data)
                    count += 1
                    if count % 100 == 0:
                        print(f'{count} / {num_data}')
                        print(f'average node number = {self.sum_nodes / count}')
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    def get_local_subgraph(self, interaction):
        x = []
        edge_index = [[], []]
        node_index_to_subgraph_node_index_dict = {}
        subgraph_node_index_to_node_dict = {}
        subgraph_interacion_nodes_index_set = set()
        subgraph_interacion_nodes_index_set.add((interaction.lncRNA.node_index, interaction.protein.node_index))
        subgraph_node_index = 0
        node_index_to_subgraph_node_index_dict[interaction.lncRNA.node_index] = subgraph_node_index
        subgraph_node_index_to_node_dict[subgraph_node_index] = interaction.lncRNA
        subgraph_node_index += 1
        node_index_to_subgraph_node_index_dict[interaction.protein.node_index] =  subgraph_node_index
        subgraph_node_index_to_node_dict[subgraph_node_index] = interaction.protein
        subgraph_node_index += 1
        for neighbor in interaction.lncRNA.interaction_list:
            interaction_set = (neighbor.lncRNA.node_index, neighbor.protein.node_index)
            if interaction_set not in self.set_interactionKey_cannot_use:
                subgraph_interacion_nodes_index_set.add(interaction_set)
                if neighbor.protein.node_index not in node_index_to_subgraph_node_index_dict.keys():
                    node_index_to_subgraph_node_index_dict[neighbor.protein.node_index] = subgraph_node_index
                    subgraph_node_index_to_node_dict[subgraph_node_index] = neighbor.protein
                    subgraph_node_index += 1
        for neighbor in interaction.protein.interaction_list:
            interaction_set = (neighbor.lncRNA.node_index, neighbor.protein.node_index)
            if interaction_set not in self.set_interactionKey_cannot_use:
                subgraph_interacion_nodes_index_set.add(interaction_set)
                if neighbor.lncRNA.node_index not in node_index_to_subgraph_node_index_dict.keys():
                    node_index_to_subgraph_node_index_dict[neighbor.lncRNA.node_index] = subgraph_node_index
                    subgraph_node_index_to_node_dict[subgraph_node_index] = neighbor.lncRNA
                    subgraph_node_index += 1
        for interaction_pair in subgraph_interacion_nodes_index_set:
            subgraph_lncRNA_index, subgraph_protein_index = node_index_to_subgraph_node_index_dict[interaction_pair[0]], \
                                                            node_index_to_subgraph_node_index_dict[interaction_pair[1]]
            edge_index[0].append(subgraph_lncRNA_index)
            edge_index[1].append(subgraph_protein_index)
            edge_index[0].append(subgraph_protein_index)
            edge_index[1].append(subgraph_lncRNA_index)
        for index, node in enumerate(subgraph_node_index_to_node_dict.values()):
            embedding_list = []
            if index == 0 or index == 1:
                embedding_list.append(0)
            else:
                embedding_list.append(1)
            for embedding in node.embedded_vector:
                embedding_list.append(float(embedding))
            x.append(embedding_list)
        if interaction.y == 1:
            y = [1]
        else:
            y = [0]
        self.sum_nodes += len(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index)
        return data

