import torch.nn.functional as F
from dataio.TemporalGraph import *
from model.TSGNet import TSGNet
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser("TSGNet",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

parser.add_argument('--numGCNlayers', default=3, type=int,
                  help='Number of GCN layers for the temporal encoder')

parser.add_argument('--numNNlayers', default=3, type=int,
                  help='Number of NN layers for the static encoder')

parser.add_argument('--sizeNNlayer', default=400, type=int,
                  help='the size of 2nd hidden layer in the static encoder')

parser.add_argument('--sizeGCNlayer', default=64, type=int,
                  help='the size of gcn in the encoder')

parser.add_argument('--epoches', default=10, type=int,
                  help='the number of epoches for training')

parser.add_argument('--nodeattributes', default=False, type=bool,
                  help='whether TSGNET uses node attributes or not.')

parser.add_argument('--dataname', default="imdb", help='data set name')

args = parser.parse_args()

# input parameters
num_GCNlayers = args.numGCNlayers
num_NNlayers = args.numNNlayers

size_hidden_GCN = args.sizeGCNlayer
size_hidden_NN = args.sizeNNlayer
num_of_epoches = args.epoches
dataset_name = args.dataname
b_use_node_attr = args.nodeattributes

if torch.cuda.is_available():
    device_info = 'cuda:0'
else:
    device_info = 'cpu'

dataset = TemporalGraph(root='/tmp/TGraph', name='imdb', tw_id='all', b_nodeattr=b_use_node_attr)
device = torch.device(device_info)
data = dataset[0].to(device)
num_tws = data.num_tws
num_nodes = data.num_nodes
num_classes = 2 # for our setting (binary classification)

static_data = data # static input

# Get temporal graph
temporal_data = [] # temporal input
for i in range(num_tws):
    dataset = TemporalGraph(root='/tmp/TGraph', name=dataset_name, tw_id=str(i), b_nodeattr=b_use_node_attr)
    device = torch.device(device_info)
    data = dataset[0].to(device)
    temporal_data.append(data)
    # in our paper, all node features are static, which means that all nodes in all time tws have the same attr.
    # just here for our convinience
    num_node_features = data.num_node_features


# Init and Train a model
model = TSGNet(num_tws, num_nodes, num_node_features, num_classes, num_GCNlayers, num_NNlayers, size_hidden_GCN, size_hidden_NN, device=device_info).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

start_time = time.time()
for epoch in range(num_of_epoches):
    optimizer.zero_grad()
    out = model(temporal_data, static_data)
    train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    print (epoch, train_loss)
    train_loss.backward()
    optimizer.step()

print("--- %s seconds ---" % (time.time() - start_time))

model.eval()
_, pred = model(temporal_data, static_data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())

print('Accuracy: {:.4f}'.format(acc))
