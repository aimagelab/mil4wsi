from models.utils.modules import FCLayer
import torch
from models.utils.basemodel import Baseline
from utils.utils2 import dropout_node


class MeanPooling(Baseline):
    def __init__(self, args,state_dict_weights):
        super(MeanPooling,self).__init__(args,state_dict_weights)

        self.mil=FCLayer(self.c_in,self.classes)


    def forward_mil(self,feats,results):
        #second step: MIL
        p_y=self.mil(feats)[1]
        results["higher"]=(p_y.mean().view(1,1),p_y.mean().view(1,1),p_y,p_y)
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

        results={}
        if self.training and self.args.dropout:
            edge_index, edge_mask, edge_node= dropout_node(edge_index=edge_index,p=0.5)
            x=x[edge_node]

        results= self.forward_mil(x,results)
        return results