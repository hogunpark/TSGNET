import collections
import os.path as osp

from torch_geometric.data import InMemoryDataset, download_url
from itertools import repeat

import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
import random
import numpy as np


class TemporalGraph(InMemoryDataset):
    def __init__(self, root, name, tw_id, b_nodeattr, transform=None, pre_transform=None):
        """Initialize TemporalGraph
            Args:
                root: path to store preprocessed inputs
                name: name of dataset
                tw_id (String): temporal id (if there exists), otherwise, "all"
                b_nodeattr: if temporal nodes use node attributes
                transform:: we do not use this
                pre_transform: we do not use this
        """
        self.name = name
        self.tw_id = tw_id
        self.b_nodeattr = b_nodeattr
        super(TemporalGraph, self).__init__(root, transform, pre_transform)

        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self): # do not use for now
        return ['some_file_1']

    @property
    def processed_file_names(self):
        return ['data_' + self.name + '_' + self.tw_id + '.pt']

    def download(self): # do not use for now
        return
        # Download to `self.raw_dir`.

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            data = self.read_TGraph_data("dataset/" + self.name , self.tw_id)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_' + self.name + '_' + self.tw_id + '.pt'))
            i += 1

    def edge_index_from_dict(self, graph_dict, num_nodes=None):
        row, col = [], []
        for key, value in graph_dict.items():
            row += repeat([key], len(value))
            col += value
        edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
        # NOTE: There are duplicated edges and self loops in the datasets. Other
        # implementations do not remove them!
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes) # WE DO NOT remove dup edges
        return edge_index

    def read_TGraph_data(self, dir, tw_id, b_sparse = True):
        """Reads temporal graph data inputs
            Arg:
                dir: path of TGraph dataset
                tw_id: a target time window id (if applicable) and "all" will use all aggregated edges
                b_sparse: Sparse Tensor option (we recommend)
            Returns:
                The processed torch.geometric.Data
        """
        # step 0: get basic info
        f = open(dir + "/graph.info", "r")
        num_tws = int(f.readline())
        num_nodes = int(f.readline())
        num_attrs = int(f.readline())
        f.close()

        self.num_tws = num_tws
        self.num_nodes = num_nodes
        if self.b_nodeattr:
            self.num_node_feat = num_attrs
        else:
            self.num_node_feat = num_nodes

        # Step 0: get labels
        f = open(dir + "/graph.lab", "r")
        l_labeled_nd = []
        y = torch.zeros([num_nodes], dtype=torch.long)
        for line in f.readlines():  # assign labels
            token = line.split("::")
            nd = int(token[0])
            lb = int(token[1])
            l_labeled_nd.append(nd)
            y[nd] = lb
        for nd in range(num_nodes):  # put indicators for nodes who do not have labels
            if nd not in l_labeled_nd:
                y[nd] = -1
        f.close()

        if tw_id == "all": # Processing aggregated static graph inputs
            # Step 1: get edges
            f = open(dir + "/graph.tedgelist", "r")

            graph = collections.defaultdict(list)
            for line in f.readlines():
                token = line.split("::")
                s = int(token[0])
                d = int(token[1])
                graph[s].append([d])
            f.close()
            edge_index = self.edge_index_from_dict(graph, num_nodes=num_nodes)

            # Step 2: get attr
            x = None
            if b_sparse:
                x_loc = [[], []]
                x_val = []
                for n, edges in graph.items():
                    for edge in edges:
                        x_loc[0].append(n)
                        x_loc[1].append(edge[0])
                        x_val.append(1)
                x_loc = torch.LongTensor(x_loc)
                x_val = torch.FloatTensor(x_val)
                x = torch.sparse.FloatTensor(x_loc, x_val, torch.Size([num_nodes, num_nodes]))
            else:
                print("[Error] not implemented.")
                exit()

        else: # Processing Temporal graph inputs
            # Step 1: get edges
            f = open(dir + "/graph_" + tw_id + ".tedgelist", "r")
            graph = collections.defaultdict(list)
            for line in f.readlines():
                token = line.split("::")
                s = int(token[0])
                d = int(token[1])
                graph[s].append([d])
            f.close()
            edge_index = self.edge_index_from_dict(graph, num_nodes=num_nodes)

            # Step 2: get attr
            x = None
            if b_sparse:
                if not self.b_nodeattr:
                    x_loc = [[], []]
                    x_val = []
                    for i in range(num_nodes):
                        x_loc[0].append(i)
                        x_loc[1].append(i)
                        x_val.append(1)
                    x_loc = torch.LongTensor(x_loc)
                    x_val = torch.FloatTensor(x_val)
                    x = torch.sparse.FloatTensor(x_loc, x_val, torch.Size([num_nodes, self.num_node_feat]))
                else:
                    f = open(dir + "/graph.attr", "r")
                    x_loc = [[], []]
                    x_val = []
                    for line in f.readlines():  # assign labels
                        token = line.split("::")
                        nd = int(token[0])
                        attr = token[1:]
                        for i, a in enumerate(attr):
                            if a != 0:  # valid
                                x_loc[0].append(nd)
                                x_loc[1].append(i)
                                x_val.append(int(float(a))) # some att has .0
                    f.close()
                    x_loc = torch.LongTensor(x_loc)
                    x_val = torch.FloatTensor(x_val)
                    x = torch.sparse.FloatTensor(x_loc, x_val, torch.Size([num_nodes, self.num_node_feat]))
                    print (num_nodes, self.num_node_feat)
            else:
                if not self.b_nodeattr:
                    x = np.identity(num_nodes)
                else:
                    f = open(dir + "/graph.attr", "r")
                    l_attr_nd = []
                    x = np.zeros([num_nodes, self.num_node_feat])
                    for line in f.readlines():  # assign labels
                        token = line.split("::")
                        nd = int(token[0])
                        attr = token[1:]
                        attr = [int(a) for a in attr]
                        x[nd] = attr
                    for nd in range(num_nodes):  # put indicators for nodes who do not have labels
                        if nd not in l_attr_nd:
                            x[nd] = [-1] * self.num_node_feat
                    f.close()
                x = torch.Tensor(x)

        # Belows are for Random cross-validation
        random.seed(24)
        random.shuffle(l_labeled_nd)



        # # to make the ratio of label to 50/50
        # l_positive = []
        # l_negative = []
        # for nd in l_labeled_nd:
        #     if y[nd] == 1:
        #         l_positive.append(nd)
        #     elif y[nd] == 0:
        #         l_negative.append(nd)
        # if len(l_positive) > len(l_negative):
        #     l_labeled_nd = l_positive[:len(l_negative)] + l_negative
        # else:
        #     l_labeled_nd = l_negative[:len(l_positive)] + l_positive

        # mask vars assignment
        # random.seed(21)
        # random.shuffle(l_labeled_nd)

        num_labeled = len(l_labeled_nd)
        train_idx = l_labeled_nd[:int(num_labeled * 0.7)]
        test_idx = l_labeled_nd[int(num_labeled * 0.7):int(num_labeled * 0.9)]
        val_idx = l_labeled_nd[int(num_labeled * 0.9):]
        train_mask = torch.Tensor([False] * num_nodes)
        test_mask = torch.Tensor([False] * num_nodes)
        val_mask = torch.Tensor([False] * num_nodes)
        train_mask = train_mask.type(torch.bool)
        test_mask = test_mask.type(torch.bool)
        val_mask = val_mask.type(torch.bool)

        for i in train_idx:
            train_mask[i] = True
        for i in test_idx:
            test_mask[i] = True
        for i in val_idx:
            val_mask[i] = True

        data = Data(x=x, edge_index=edge_index, y=y, num_tws=self.num_tws, num_nodes=self.num_nodes)
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.val_mask = val_mask

        self.data, self.slices = self.collate([data])

        return data

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_' + self.name + '_' + self.tw_id + '.pt'))
        return data