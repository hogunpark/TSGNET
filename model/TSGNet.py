import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

class TSGNet(torch.nn.Module):
    """
    NN Module for TSGNet which includes temporal encoder and static encoder
    """

    def __init__(self, num_tws, num_nodes, num_node_features, num_classes, num_GCNlayers, num_NNlayers, size_hidden_GCN, size_hidden_NN, device="cpu"):
        """
        We note that we have one layer for LSTM and final NN layer

        :param num_tws:  a total number of time windoews
        :param num_nodes: a number of nodes in the dataset
        :param num_node_features: a number of node features in the dataset
        :param num_classes: a number of classes in the dataset
        :param num_GCNlayers: a number of gcn layers in temporal encoder
        :param num_NNlayers:  a number of nn layers in statis encoder
        :param size_hidden_GCN: the size of hidden node in GCNs of temporal encoder
        :param size_hidden_NN: the size of 1st nn layer of static encoder
        """
        super(TSGNet, self).__init__()

        self.num_nodes = num_nodes
        self.num_node_feat = num_node_features
        self.num_classes = num_classes

        self.size_hidden_GCNs = size_hidden_GCN
        self.device = device

        if num_GCNlayers < 2 or num_NNlayers < 2:  # internal recommendation
            print("[Error] # of GCN OR NN layers should be more than 1")
            exit()

        """
        Step 1: init GCNs for temporal encoders
        """
        self.l_temp_gcns = []

        for tw in range(num_tws):
            l_temp = []
            conv1 = GCNConv(self.num_node_feat, size_hidden_GCN, bias=False)  # layer 1
            l_temp.append(conv1)
            for tw_remaining in range(num_GCNlayers - 1):  # layer 2 -
                conv2 = GCNConv(size_hidden_GCN, size_hidden_GCN, bias=False)
                l_temp.append(conv2)
            self.l_temp_gcns.append(torch.nn.ModuleList(l_temp))
        self.l_temp_gcns = torch.nn.ModuleList(self.l_temp_gcns)

        """
        Step 2: init LSTM
        """

        self.lstm = torch.nn.LSTM(size_hidden_GCN, size_hidden_GCN, 1)  # 1 LSTM layer

        """
        Step 3: init NN for Static encoder
        """
        self.l_static_nn = []
        self.l_static_nn.append(torch.nn.Linear(num_nodes, size_hidden_NN))  # 1st NN later

        for s_remaining in range(num_NNlayers - 2):  # layer 2 -
            self.l_static_nn.append(torch.nn.Linear(size_hidden_NN, int(size_hidden_NN / 4)))
            size_hidden_NN = int(size_hidden_NN / 4)
        self.l_static_nn.append(torch.nn.Linear(size_hidden_NN, size_hidden_GCN))  # last layer in static encoder
        self.l_static_nn = torch.nn.ModuleList(self.l_static_nn)

        """
        Step 4: final NN
        """
        self.final_nn = torch.nn.Linear(size_hidden_GCN, num_classes)

    def forward(self, t_data, s_data):
        """ Forward messages

        :param t_data: temporal graphs, list of data
        :param s_data: statis graph, a data
        :return: outputs of a TSGNET
        """

        # Step 1: Temporal encoder
        l_temporal_output = []
        for t_idx, t_graph in enumerate(t_data):  # each temporal graph has own gcn filter and inputs
            t, t_edge_index = t_graph.x, t_graph.edge_index
            l_conv = self.l_temp_gcns[t_idx]
            for g_idx, conv in enumerate(l_conv):
                t = conv(t, t_edge_index)
                if g_idx == 0:
                    t = F.relu(t)
                t = F.dropout(t, training=self.training, p=0.5)
            l_temporal_output.append(t)

        t = torch.stack(l_temporal_output, dim=0)  # dim: [|T|, |V|, H], where H is size of GCN hidden layer

        # Step 2: Static encoder
        x, s_edge_index = s_data.x, s_data.edge_index
        for s_idx, s_layer in enumerate(self.l_static_nn):
            if s_idx == 0:
                x = s_layer(x)
                x = F.relu(x)
            else:
                x = s_layer(x)

        x = F.log_softmax(x, dim=1)


        # Step 3: LSTM layer after GCNs
        if self.device == "cpu":
            init_lstm = (torch.randn((1, self.num_nodes, self.size_hidden_GCNs)),
                         torch.randn((1, self.num_nodes, self.size_hidden_GCNs)))  # 1: num of lstm layers
        else:
            init_lstm = (torch.randn((1, self.num_nodes, self.size_hidden_GCNs)).cuda(),
                     torch.randn((1, self.num_nodes, self.size_hidden_GCNs)).cuda())  # 1: num of lstm layers


        z, _ = self.lstm(t, init_lstm)  # (seq_len, batch, input_size)
        z = z[-1]  # we use the last output
        z = F.log_softmax(z, dim=1)

        # Step 4: Temporal + Static
        h = self.final_nn(z + x)
        # h = F.relu(h)
        # h = F.dropout(h, training=self.training, p=0.5)

        return F.log_softmax(h, dim=1)
        # return F.log_softmax(h, dim=1)