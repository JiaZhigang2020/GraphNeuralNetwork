import os
from generate_edgelist import read_interactioNPInter2
from classes import LncRNAProteinInteraction
import argparse
from methods import get_node_index_to_index_dict
import random
import os.path as osp
from classes import MyOwnDataset


def parase_args():
    parser = argparse.ArgumentParser(description="Generate dataset.")
    parser.add_argument('--fold', default=0, help="int, which fold is this.")
    return parser.parse_args()


def read_interaction_nodes_index(interaction_nodes_index_file_path):
    interaction_nodes_set = set()
    with open(interaction_nodes_index_file_path, 'r') as reader:
        file_content = reader.readlines()
    for single_line in file_content:
        interaction_nodes_list = single_line.split(",")
        interaction_nodes_set.add((int(interaction_nodes_list[0]), int(interaction_nodes_list[1][:-1])))
    return interaction_nodes_set


def node_index_to_node_object_dict(node_object_list):
    node_index_to_index_dict = {}
    for node_object in node_object_list:
        node_index_to_index_dict[node_object.node_index] = node_object
    return node_index_to_index_dict


def rebuild_negativeInteraction(negative_interaction_nodes_index_set, lncRNA_list, protein_list, negative_interaction_list):
    node_index_to_lncRNA_object_dict = node_index_to_node_object_dict(lncRNA_list)
    node_index_to_protein_object_dict = node_index_to_node_object_dict(protein_list)
    for negative_interaction_node_index_set in negative_interaction_nodes_index_set:
        # print(node_index_to_lncRNA_object_dict)
        # print(negative_interaction_node_index_set)
        lncRNA = node_index_to_lncRNA_object_dict[negative_interaction_node_index_set[0]]
        protein = node_index_to_protein_object_dict[negative_interaction_node_index_set[1]]
        negative_interaction = LncRNAProteinInteraction(lncRNA, protein, 0, negative_interaction_node_index_set)
        negative_interaction_list.append(negative_interaction)
        lncRNA.interaction_list.append(negative_interaction)
        protein.interaction_list.append(negative_interaction)
    return negative_interaction_nodes_index_set, lncRNA_list, protein_list, negative_interaction_list


def get_nodes_list_and_edge_list(lncRNA_list, protein_list, positive_interaction_nodes_index_set,
                                   negative_interaction_nodes_index_set):
    node_list = lncRNA_list + protein_list
    edge_list = list(positive_interaction_nodes_index_set) + list(negative_interaction_nodes_index_set)
    return node_list, edge_list


def read_node2vec(node2vec_encode_file_path, lncRNA_list, protein_list, positive_interaction_nodes_index_set,
                  negative_interaction_nodes_index_set):
    node_list, edge_list = get_nodes_list_and_edge_list(lncRNA_list, protein_list, positive_interaction_nodes_index_set,
                                                        negative_interaction_nodes_index_set)
    node_index_to_index_dict = get_node_index_to_index_dict(node_list)
    with open(node2vec_encode_file_path, 'r', encoding='utf-8') as reader:
        file_content = reader.readlines()
        file_content.pop(0)
        for single_line in file_content:
            encode_list = single_line.strip().split(' ')
            node_index = int(encode_list[0])
            encode_list.pop(0)
            node_list[node_index_to_index_dict[node_index]].embedded_vector = encode_list
        count_node_without_node2vecResult = 0
        for node in node_list:
            if len(node.embedded_vector) != 64:
                count_node_without_node2vecResult += 1
                node.embedded_vector = [0] * 64
        print(f"没有node2vec结果的节点数：{count_node_without_node2vecResult}")
    return lncRNA_list, protein_list, positive_interaction_nodes_index_set, negative_interaction_nodes_index_set


if __name__ == '__main__':
    args = parase_args()
    interaction_file_path = f'./data/NPInter2.xlsx'
    positive_interaction_list, negative_interaction_list, lncRNA_list, protein_list, lncRNA_name_index_dict, \
    protein_name_index_dict, positive_interaction_nodes_index_set, negative_interaction_nodes_index_set = \
        read_interactioNPInter2(dataset_path=interaction_file_path, dataset_name="NPInter2")
    interaction_nodes_index_file_path = "./data/negative_interaction_nodes_index_all.edge"
    negative_interaction_nodes_index_set = read_interaction_nodes_index(interaction_nodes_index_file_path=interaction_nodes_index_file_path)
    negative_interaction_nodes_index_set, lncRNA_list, protein_list, negative_interaction_list = \
        rebuild_negativeInteraction(negative_interaction_nodes_index_set, lncRNA_list, protein_list, negative_interaction_list)
    negative_interaction_nodes_index_file_train_path = f"./data/train/negative_interaction_nodes_index_train_fold{args.fold}.edge"
    positive_interaction_nodes_index_file_train_path = f"./data/train/positive_interaction_nodes_index_train_fold{args.fold}.edge"
    negative_interaction_nodes_index_file_test_path = f"./data/test/negative_interaction_nodes_index_test_fold{args.fold}.edge"
    positive_interaction_nodes_index_file_test_path = f"./data/test/positive_interaction_nodes_index_test_fold{args.fold}.edge"
    negative_train_interaction_nodes_index_set = read_interaction_nodes_index(interaction_nodes_index_file_path=negative_interaction_nodes_index_file_train_path)
    positive_train_interaction_nodes_index_set = read_interaction_nodes_index(interaction_nodes_index_file_path=positive_interaction_nodes_index_file_train_path)
    negative_test_interaction_nodes_index_set = read_interaction_nodes_index(interaction_nodes_index_file_path=negative_interaction_nodes_index_file_test_path)
    positive_test_interaction_nodes_index_set = read_interaction_nodes_index(interaction_nodes_index_file_path=positive_interaction_nodes_index_file_test_path)
    node2vec_encode_file_path = f"./data/node2vec_result/training_{args.fold}/result.emb"
    lncRNA_list, protein_list, positive_interaction_nodes_index_set, negative_interaction_nodes_index_set = \
        read_node2vec(node2vec_encode_file_path, lncRNA_list, protein_list, positive_interaction_nodes_index_set,
                  negative_interaction_nodes_index_set)
    all_interaction_list = positive_interaction_list + negative_interaction_list
    random.shuffle(all_interaction_list)
    set_interactionKey_cannot_use = set()
    set_interactionKey_cannot_use.update(positive_test_interaction_nodes_index_set)
    set_interactionKey_cannot_use.update(negative_test_interaction_nodes_index_set)
    dataset_train_path = f'./data/inMemory_train_0/'
    if not osp.exists(dataset_train_path):
        os.makedirs(dataset_train_path)
        print(f"创建了文件夹：{dataset_train_path}")
    interaction_for_generate = set()
    interaction_for_generate.update(positive_train_interaction_nodes_index_set)
    interaction_for_generate.update(negative_train_interaction_nodes_index_set)
    My_trainingDataset = MyOwnDataset(dataset_train_path, interaction_list=all_interaction_list, set_interactionKey_can_use=interaction_for_generate,
                                      set_interactionKey_cannot_use=set_interactionKey_cannot_use)
    dataset_test_path = f'./data/inMemory_test_0/'
    if not osp.exists(dataset_test_path):
        os.makedirs(dataset_test_path)
        print(f"创建了文件夹：{dataset_test_path}")
    interaction_for_generate = set()
    interaction_for_generate.update(positive_test_interaction_nodes_index_set)
    interaction_for_generate.update(negative_test_interaction_nodes_index_set)
    My_trainingDataset = MyOwnDataset(dataset_test_path, interaction_list=all_interaction_list,
                                      set_interactionKey_can_use=interaction_for_generate,
                                      set_interactionKey_cannot_use=set_interactionKey_cannot_use)


