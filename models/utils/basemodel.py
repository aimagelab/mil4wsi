import torch.nn as nn
import torch
import wandb
from utils.losses import *
from utils.utils2 import dropout_node


class Baseline(nn.Module):

    def __init__(self, args,state_dict_weights=None):
        """
        Inputs:
            args: NameSpace
        """
        super().__init__()
        self.args=args
        self.target= args.target
        self.lamb=args.lamb
        self.beta=args.beta
        self.dropout= args.dropout
        self.tau=args.temperature
        self.kl= args.kl
        self.residual=args.residual
        self.c_in=args.input_size
        self.classes=args.n_classes
        self.c_hidden=args.c_hidden
        self.add_bias=args.add_bias
        self.betapreg=args.preg
        self.max=args.max

        self.state_dict_weights=state_dict_weights


    def forward_scale(self,x: torch.Tensor,edge_index: torch.Tensor,gnnlayer: torch.nn.Module)-> tuple[torch.Tensor,torch.Tensor]:
        """forward scale into gnn module

        Args:
            x (torch.Tensor): input
            edge_index (torch.Tensor): adjency matrix
            gnnlayer (torch.nn.Module): module

        Returns:
            tuple[torch.Tensor,torch.Tensor]: return output and new adiency matrix
        """
        r=x
        if self.training and self.args.dropout:
            edge_index, _, _= dropout_node(edge_index=edge_index)
        x=gnnlayer(x,edge_index)
        if self.residual:
            x=r+x
        else:
            x=x
        return x,edge_index

    def forward_gnn(self,x: torch.Tensor,edge_index: torch.Tensor,levels: torch.Tensor,childof: torch.Tensor,edge_index2: torch.Tensor=None,edge_index3 : torch.Tensor=None):
        NotImplementedError("forward_gnn error")

    def forward_mil(self,indecesperlevel: torch.Tensor,feats: torch.Tensor,results: dict):
        NotImplementedError("forward_mil error")

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
        feats,indecesperlevel,results=self.forward_gnn(x, edge_index, levels, childof, edge_index2, edge_index3)
        results= self.forward_mil(indecesperlevel,feats,results)
        return results


    def compute_loss(self,loss_module_instance: torch.nn.Module,results:dict,bag_label:torch.Tensor)-> torch.Tensor:
        """compute loss

        Args:
            loss_module_instance (torch.Module):  loss criterion
            results (dict): model output
            bag_label (torch.Tensor): GT

        Returns:
            torch.Tensor: return loss
        """
        loss=0
        #target level prediction
        y_instance_pred1,prediction_bag1,_,_=results[self.target]
        higher_loss= CELoss(loss_module_instance,y_instance_pred1,prediction_bag1,bag_label)
        loss+=higher_loss
        #------
        if self.kl is not None:
            #second scale loss
            y_instance_pred2,prediction_bag2,_,_=results[self.kl]
            lower_loss=CELoss(loss_module_instance,y_instance_pred2,prediction_bag2,bag_label,lamb=(1.0-self.lamb))
            loss+=lower_loss
            #-----
            #distill patch level
            if self.max:
                mse_loss=computeMSE_max(y_instance_pred2,y_instance_pred1,results["childof"])
            else:
                mse_loss=computeMSE(y_instance_pred2,y_instance_pred1,results["childof"])
            loss+=self.beta*mse_loss
            #------
            kl_bag_loss= computeKL(prediction_bag2,prediction_bag1,self.tau,self.classes,self.add_bias)
            loss+= self.lamb*kl_bag_loss
            #-------
        return loss


    def predict(self,results):
        prediction_patch_higher,prediction_bag_higher,_,_=results[self.target]
        if self.classes>1:
            higher_prediction=torch.sigmoid(prediction_bag_higher)
            if self.kl is not None:
                _,prediction_bag_lower,A,B=results[self.kl]
                lower_prediction=0.5*torch.sigmoid(prediction_bag_lower)+0.5 *torch.sigmoid(prediction_bag_higher)[:,-1]
            else:
                lower_prediction=None
        else:
            higher_prediction=0.5*torch.sigmoid(prediction_bag_higher)[:,-1]+0.5 *torch.sigmoid(prediction_bag_higher)[:,-1]
            if self.kl is not None:
                _,prediction_bag_lower,_,_=results[self.kl]
                lower_prediction=torch.sigmoid(prediction_bag_lower)[:,-1]
            else:
                lower_prediction=None
        return higher_prediction,lower_prediction

