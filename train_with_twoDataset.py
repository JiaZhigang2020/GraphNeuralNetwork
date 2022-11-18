import argparse
import os.path as osp
import os
import torch
from torch_geometric.nn import GCNConv, TopKPooling, SAGEConv, EdgeConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from generate_dataset import MyOwnDataset
from torch_geometric.loader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument("--fold", default=5, help="Fold of training.")
    parser.add_argument("--batch_size", default=200, help="How many sample do you want to calculate at the same time.")
    return parser.parse_args()


class Model(torch.nn.Module):
    def __init__(self, num_node_features):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(num_node_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.5)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.5)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.5)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print(x.shape, edge_index.shape, batch)
        input()
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = torch.sigmoid(x)
        return x


def train(model, train_dataloader, optimizer, device):
    model.train()
    loss_all = 0
    for data in train_dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = data.y.type(torch.FloatTensor).view(len(data.y), 1).to(device)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        loss_all += len(data) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(model, test_dataloader, device):
    model.eval()
    loss_all = 0
    for data in test_dataloader:
        data = data.to(device)
        output = model(data)
        target = data.y.type(torch.FloatTensor).view(len(data.y), 1).to(device)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        loss_all += len(data) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


if __name__ == '__main__':
    args = parse_args()
    training_dataset_path = f'./data/inMemory_train_0/'
    testing_datast_path = f'./data/inMemory_test_0/'
    train_dataset = MyOwnDataset(root=training_dataset_path).shuffle()
    test_dataset = MyOwnDataset(root=testing_datast_path).shuffle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saving_floder_path = 'result/model/'
    if not osp.exists(saving_floder_path):
        os.makedirs(saving_floder_path)
        print(f'create "{saving_floder_path}" floder')
    num_of_epoches = 100
    lr = 0.001
    l2_weight_decay = 0.001
    model = Model(train_dataset.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    loss_last = float('inf')
    for epoch in range(num_of_epoches):
        loss = train(model, train_dataloader, optimizer, device)
        if loss > loss_last:
            scheduler.step()
        loss_last = loss
        test_loss = test(model, test_dataloader, device)
        print(f"epoch:{epoch}, train loss: {loss}, test loss: {test_loss}")






