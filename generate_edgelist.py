import argparse
import os
import random
import copy
import os.path as osp
import pandas as pd
import networkx as nx
from classes import LncRNA, Protein, LncRNAProteinInteraction


def parse_arguments():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument("--interactionDatasetName", default="NPInter2", help="raw interactions dataset")
    parser.add_argument("--projectName", default="NPI_recurrence", help="projectName")
    return parser.parse_args()


def read_interactioNPInter2(dataset_path, dataset_name):
    positive_interaction_list = [] # 放置所有的正样本的互作关系（i.e. 存在互作关系）， （lncRNA对象, portein对象, label, interaction_nodes_index_set(e.g. set格式的lncRNA对象的node_index和protein对象的nodex_index， nodex_index是图里所有的lncRNA和protein对象作为图顶点的编码)）
    negative_interaction_list = [] # 放置所有的负样本的互作关系（i.e. 不存在互作关系）， （lncRNA对象, portein对象, label, interaction_nodes_index_set(e.g. set格式的lncRNA对象的node_index和protein对象的nodex_index， nodex_index是图里所有的lncRNA和protein对象作为图顶点的编码)）
    lncRNA_list = [] # 所有的lncRNA对象，无重复
    protein_list = [] # 所有的protein对象，无重复
    lncRNA_name_index_dict = {} # 键值对， lncRNA_name: index, 注意这里的index是表示lncRNA在lncRNAlist里存储的index，是为了方便查找lncRNA对象在lncRNA_list里的index，而不是lncRNA对象里存储的node_index
    protein_name_index_dict = {} # 键值对， protein_name: index， 注意这里的index是表示protein在lncRNAlist里存储的index，是为了方便查找protein对象在protein_list里的index，而不是lncRNA对象里存储的node_index
    positive_interaction_nodes_index_set = set() # set格式存储的所有正样本的interaction_nodes_index， interaction_nodes_index里存储的是set格式的lncRNA和portein的nodes_index
    negative_interaction_nodes_index_set = set() # set格式存储的所有负样本的interaction_nodes_index， interaction_nodes_index里存储的是set格式的lncRNA和portein的nodes_index
    if not osp.exists(dataset_path):
        raise Exception("interaction dataset does does not exist")
    original_data_pd = pd.read_excel(io=dataset_path, sheet_name="Sheet1")
    node_index, lncRNA_index, protein_index = 0, 0, 0
    for index, (lncRNA_name, protein_name, label) in original_data_pd.iterrows():
        if lncRNA_name not in lncRNA_name_index_dict:
            lncRNA = LncRNA(name=lncRNA_name, node_index=node_index, node_type="LncRNA")
            lncRNA_list.append(lncRNA)
            lncRNA_name_index_dict[lncRNA_name] = lncRNA_index
            node_index += 1
            lncRNA_index += 1
        else:
            lncRNA = lncRNA_list[lncRNA_name_index_dict[lncRNA_name]]
        if protein_name not in protein_name_index_dict:
            protein = Protein(name=protein_name, node_index=node_index, node_type="Protein")
            protein_list.append(protein)
            protein_name_index_dict[protein_name] = protein_index
            node_index += 1
            protein_index += 1
        else:
            protein = protein_list[protein_name_index_dict[protein_name]]
        interaction_nodes_index_set = (lncRNA.node_index, protein.node_index)
        interaction = LncRNAProteinInteraction(lncRNA=lncRNA, protein=protein, y=label, interaction_nodes_index_set=interaction_nodes_index_set)
        lncRNA.interaction_list.append(interaction)
        protein.interaction_list.append(interaction)

        if label == 1:
            positive_interaction_list.append(interaction)
            positive_interaction_nodes_index_set.add(interaction_nodes_index_set)
        elif label == 0:
            negative_interaction_list.append(interaction)
            negative_interaction_nodes_index_set.add(interaction_nodes_index_set)
        else:
            raise Exception(f"{dataset_name} has labels other than 0 and 1. label: {label}")
    print("number of lncRNA: {}, number of protein: {}, number of node: {}".format(lncRNA_index, protein_index,
                                                                                   lncRNA_index + protein_index))
    print("number of positive interaction: {}, number of negative interaction: {}, "
          "number of interation: {}".format(len(positive_interaction_list), len(negative_interaction_list),
                                            len(positive_interaction_list) + len(negative_interaction_list)))
    return positive_interaction_list, negative_interaction_list, lncRNA_list, protein_list, lncRNA_name_index_dict, \
           protein_name_index_dict, positive_interaction_nodes_index_set, negative_interaction_nodes_index_set


def negative_interaction_generation(positive_interaction_list, negative_interaction_list, lncRNA_list, protein_list,
                                    lncRNA_name_index_dict, protein_name_index_dict,
                                    positive_interaction_nodes_index_set, negative_interaction_nodes_index_set):
    if len(negative_interaction_list) != 0:
        raise Exception("negative interaction exist.")
    num_of_positive_interaction = len(positive_interaction_list)
    num_of_lncRNA = len(lncRNA_list)
    num_of_protein = len(protein_list)
    num_of_negative_interaction = 0
    while(num_of_negative_interaction < num_of_positive_interaction):
        random_index_lncRNA = random.randint(0, num_of_lncRNA - 1)
        random_index_protein = random.randint(0, num_of_protein -1)
        selected_lncRNA = lncRNA_list[random_index_lncRNA]
        selected_protein = protein_list[random_index_protein]
        interaction_nodes_index_set = (selected_lncRNA.node_index, selected_protein.node_index)
        if interaction_nodes_index_set in positive_interaction_nodes_index_set:
            continue
        if interaction_nodes_index_set in negative_interaction_nodes_index_set:
            continue
        negative_interaction_nodes_index_set.add(interaction_nodes_index_set)
        interaction = LncRNAProteinInteraction(selected_lncRNA, selected_protein, 0, interaction_nodes_index_set)
        negative_interaction_list.append(interaction)
        selected_lncRNA.interaction_list.append(interaction)
        selected_protein.interaction_list.append(interaction)
        num_of_negative_interaction += 1
    print(f'generate {len(negative_interaction_list)} negative samples.')
    return  positive_interaction_list, negative_interaction_list, lncRNA_list, protein_list, lncRNA_name_index_dict, \
    protein_name_index_dict, positive_interaction_nodes_index_set, negative_interaction_nodes_index_set


def networkx_format_generation(positive_interaction_list, negative_interaction_list, lncRNA_list, protein_list):
    edge_list = positive_interaction_list + negative_interaction_list
    node_list = lncRNA_list + protein_list
    G = nx.Graph()
    for node in node_list:
        G.add_node(node.node_index)
    for edge in edge_list:
        G.add_edge(edge.lncRNA.node_index, edge.protein.node_index)
    print(f"number of nodes in graph: {G.number_of_nodes()}, number of edges in graph: {G.number_of_edges()}")
    print(f"number of connected componet: {len(list(nx.connected_components(G)))}")
    return G


def save_edgelist_file(G, output_floder_path):
    if not osp.exists(output_floder_path):
        os.makedirs(output_floder_path)
        print(f'创建了文件， {output_floder_path}')
    output_file_path = output_floder_path + './bipartite_graph.edgelist'
    if osp.exists(output_file_path):
        is_rewrite = input('edgelist file already exist, rewrite or not? Y/N')[0]
        if is_rewrite in "Nn":
            exit()
    nx.write_edgelist(G, path=output_file_path)


def save_interaction_nodes_index(save_file_path:str, interaction_nodes_index_set:set):
    print(f"保存： {save_file_path}")
    save_floder_path = osp.dirname(save_file_path)
    if not osp.exists(save_floder_path):
        print(f"创建了文件： {save_floder_path}")
        os.makedirs(save_floder_path)
    with open(save_file_path, mode='w', encoding='utf-8') as writer:
        for interaction_nodes_index in interaction_nodes_index_set:
            writer.write(f'{interaction_nodes_index[0]}, {interaction_nodes_index[1]}\n')


def generate_G_training(G, positive_interaction_nodes_index_test_set, negative_interaction_nodes_index_test_set, fold_number):
    G_training = copy.deepcopy(G)
    for positive_interaction_nodes_index_test in positive_interaction_nodes_index_test_set:
        G_training.remove_edge(*positive_interaction_nodes_index_test)
    for negative_interaction_nodes_index_test in negative_interaction_nodes_index_test_set:
        G_training.remove_edge(*negative_interaction_nodes_index_test)
    print(f'{fold_number} fold training dataset graph: number of nodes: {G_training.number_of_nodes()}, number of edges: {G_training.number_of_edges()}')
    print(f'number of connected components: {len(list(nx.connected_components(G_training)))}')
    save_edgelist_file(G_training, f'data/graph/training_{fold_number}/')


def generate_training_and_testing_data(G, positive_interaction_nodes_index_set, negative_interaction_nodes_index_set, save_floder_path="./data/"):
    positive_interaction_split_list = [set(), set(), set(), set(), set()]
    negative_interaction_split_list = [set(), set(), set(), set(), set()]
    count = 0
    while len(positive_interaction_nodes_index_set) > 0:
        positive_interaction_split_list[count % 5].add(positive_interaction_nodes_index_set.pop())
        count += 1
    while len(negative_interaction_nodes_index_set) > 0:
        negative_interaction_split_list[count % 5].add(negative_interaction_nodes_index_set.pop())
        count += 1
    for i in range(5):
        positive_interaction_nodes_index_train_set = set()
        negative_interaction_nodes_index_train_set = set()
        positive_interaction_nodes_index_test_set = set()
        negative_interaction_nodes_index_test_set = set()
        for j in range(5):
            if i == j:
                positive_interaction_nodes_index_test_set.update(positive_interaction_split_list[j])
                negative_interaction_nodes_index_test_set.update(negative_interaction_split_list[j])
            else:
                positive_interaction_nodes_index_train_set.update(positive_interaction_split_list[j])
                negative_interaction_nodes_index_train_set.update(negative_interaction_split_list[j])
        save_interaction_nodes_index(
            save_file_path=save_floder_path + f"test/positive_interaction_nodes_index_test_fold{i}.edge",
            interaction_nodes_index_set=positive_interaction_nodes_index_test_set)
        save_interaction_nodes_index(
            save_file_path=save_floder_path + f"test/negative_interaction_nodes_index_test_fold{i}.edge",
            interaction_nodes_index_set=negative_interaction_nodes_index_test_set)
        save_interaction_nodes_index(
            save_file_path=save_floder_path + f"train/positive_interaction_nodes_index_train_fold{i}.edge",
            interaction_nodes_index_set=positive_interaction_nodes_index_train_set)
        save_interaction_nodes_index(
            save_file_path=save_floder_path + f"train/negative_interaction_nodes_index_train_fold{i}.edge",
            interaction_nodes_index_set=negative_interaction_nodes_index_train_set)
        generate_G_training(G, positive_interaction_nodes_index_test_set, negative_interaction_nodes_index_test_set,
                            fold_number=i)


def create_node2vec_result_floder(save_floder_path="./data/node2vec_result/"):
    if not osp.exists(save_floder_path):
        os.makedirs(save_floder_path)
        print(f"创建了node2vec的保存文件夹： {save_floder_path}")
    for i in range(5):
        node2vec_per_flod_path = f"{save_floder_path}training_{i}/"
        if not osp.exists(node2vec_per_flod_path):
            os.makedirs(node2vec_per_flod_path)
            print(f"创建了node2vec的保存文件夹： {node2vec_per_flod_path}")


if __name__ == '__main__':
    args = parse_arguments()
    interaction_dataset_path = 'data/' + args.interactionDatasetName + '.xlsx'
    positive_interaction_list, negative_interaction_list, lncRNA_list, protein_list, lncRNA_name_index_dict, \
    protein_name_index_dict, positive_interaction_nodes_index_set, negative_interaction_nodes_index_set = \
        read_interactioNPInter2(dataset_path=interaction_dataset_path, dataset_name=args.interactionDatasetName)
    positive_interaction_list, negative_interaction_list, lncRNA_list, protein_list, lncRNA_name_index_dict, \
    protein_name_index_dict, positive_interaction_nodes_index_set, negative_interaction_nodes_index_set = \
        negative_interaction_generation( positive_interaction_list, negative_interaction_list, lncRNA_list,
                                         protein_list, lncRNA_name_index_dict, protein_name_index_dict,
                                         positive_interaction_nodes_index_set, negative_interaction_nodes_index_set)
    save_interaction_nodes_index(save_file_path='data/negative_interaction_nodes_index_all.edge',
                                 interaction_nodes_index_set=negative_interaction_nodes_index_set)
    G = networkx_format_generation(positive_interaction_list, negative_interaction_list, lncRNA_list, protein_list)
    graph_save_floder_path = f'data/graph/{args.projectName}'
    save_edgelist_file(G=G, output_floder_path=graph_save_floder_path)
    generate_training_and_testing_data(G, positive_interaction_nodes_index_set, negative_interaction_nodes_index_set)
    create_node2vec_result_floder(save_floder_path='./data/node2vec_result/')
