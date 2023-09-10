from models.utils.modules import FCLayer,BClassifier,MILNet,init
import torch
from models.utils.basemodel import Baseline
from utilsmil4wsi.utils2 import dropout_node


class DSMIL(Baseline):
    def __init__(self, args,state_dict_weights):
        super(DSMIL,self).__init__(args,state_dict_weights)

        milfc,milbag=FCLayer(self.c_in,self.classes),BClassifier(self.c_in,self.classes)
        self.mil=MILNet(milfc,milbag)
        self.mil=init(self.mil,self.state_dict_weights)

    def forward_mil(self,feats,results):
        #second step: MIL
        results["higher"]=self.mil(feats)#x5x20
        return results

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,levels: torch.Tensor,childof: torch.Tensor,edge_index2: torch.Tensor=None,edge_index3: torch.Tensor=None):
        """forward model

        Args:
            x (torch.Tensor): input
            edge_index (torch.Tensor): adjecency matrix
            levels (torch.Tensor): scale level
            childof (torch.Tensor): interscale information
            edge_index2 (torch.Tensor, optional): adjecency matrix of single scale
            edge_index3 (torch.Tensor, optional): adjecency matrix of single scale

        Returns:
            _type_: _description_
        """
        #feats,indecesperlevel,results=self.forward_gnn(x, edge_index, levels, childof, edge_index2, edge_index3)
        if self.training and self.args.dropout:
            edge_index, edge_mask, edge_node= dropout_node(edge_index=edge_index,p=0.5)
            x=x[edge_node]
        results={}
        results= self.forward_mil(x,results)
        return results