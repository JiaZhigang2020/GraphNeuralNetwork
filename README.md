# GraphNeuralNetwork
这项工作是对NPI-GNN工作的复现
NPI-GNN项目地址：https://github.com/AshuiRUA/NPI-GNN
NPI-GNN论文Pubmed链接：https://pubmed.ncbi.nlm.nih.gov/33822882/

1.生成edgelist，划分训练集/测试集

python generate_edgelist.py


2.生成node2vec编码

Python .\node2vec-master\src\main.py --input 'data/graph/training_{0-4}/bipartite_graph.edgelist' --output 'data/node2vec_result/training_{0-4}/result.emb'


3.生成dataset

python generate_dataset.py


4.训练模型 （由于GNN和CNN模型评价方面并没有很大不同，所以模型评估中只给出了loss）

python train_with_twoDataset.py
