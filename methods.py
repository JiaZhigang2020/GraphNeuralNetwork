def get_node_index_to_index_dict(node_list):
    nodes_index_to_index_dict = {}
    for index in range(len(node_list)):
        nodes_index_to_index_dict[node_list[index].node_index] = index
    return nodes_index_to_index_dict
