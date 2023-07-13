from models.utils.modules import GNNModel, FCLayer, BClassifier, MILNet, init
from models.utils.basemodel import Baseline
import torch


class DASMIL(Baseline):
    def __init__(self, args, state_dict_weights):
        """
        Inputs:
            args: DASMIL model implementation
        """
        super(DASMIL, self).__init__(args, state_dict_weights)
        self.GNNlast = GNNModel(c_in=self.c_in,
                                c_hidden=self.c_hidden,
                                c_out=self.c_hidden,
                                num_layers=args.num_layers, layer_name=args.layer_name, dp_rate=args.dropout_rate, heads=args.heads)
        self.gnn1 = GNNModel(c_in=self.c_in,
                             c_hidden=self.c_hidden,
                             c_out=self.c_in,
                             num_layers=args.num_layers, layer_name=args.layer_name, dp_rate=args.dropout_rate, heads=args.heads)
        self.gnn2 = GNNModel(c_in=self.c_in,
                             c_hidden=self.c_hidden,
                             c_out=self.c_in,
                             num_layers=args.num_layers, layer_name=args.layer_name, dp_rate=args.dropout_rate, heads=args.heads)

        mil2fc, mil2bag = FCLayer(self.c_hidden, self.classes), BClassifier(
            self.c_hidden, self.classes)
        self.mil2 = MILNet(mil2fc, mil2bag)
        mil3fc, mil3bag = FCLayer(self.c_hidden, self.classes), BClassifier(
            self.c_hidden, self.classes)
        self.mil3 = MILNet(mil3fc, mil3bag)
        self.mil2 = init(self.mil2, self.state_dict_weights)
        self.mil3 = init(self.mil3, self.state_dict_weights)

    def forward_gnn(self, x, edge_index, levels, childof, edge_index2=None, edge_index3=None):
        results = {}
        featsperlevel = []
        indecesperlevel = []
        # forward input for each scale gnn
        for i in levels.unique():
            # select scale
            indeces_feats = (levels == i).nonzero().view(-1)
            feats = x[indeces_feats]
            if i == levels.min():
                feats, _ = self.forward_scale(feats, edge_index2, self.gnn1)
            else:
                feats, _ = self.forward_scale(feats, edge_index3, self.gnn2)
            featsperlevel.append(feats)
            indecesperlevel.append(indeces_feats)
        # merge scales
        feats = torch.concat(featsperlevel)  # [N+M,250]
        indeces = torch.concat(indecesperlevel).view(-1)  # [N+M,]
        indeces = indeces.sort()[1]
        feats = feats[indeces]
        # forward togheter
        feats, edge_index2 = self.forward_scale(
            feats, edge_index, self.GNNlast)
        results["childof"] = childof[indecesperlevel[1]]
        # -------------------------------------------------------------------------------
        return feats, indecesperlevel, results

    def forward_mil(self, indecesperlevel, feats, results):
        # second step: MIL
        results["lower"] = self.mil2(
            feats[results["childof"].unique()].view(-1, self.c_hidden))  # x5
        results["higher"] = self.mil3(
            feats[indecesperlevel[1]].view(-1, self.c_hidden))  # x20
        return results
