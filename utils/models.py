import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch
import torch.nn.functional as F
from utils.utils2 import dropout_node

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GATV2": geom_nn.GATv2Conv,
    "GAT":geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

class GNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden

        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             heads=3,
                             concat=False,
                             dropout=0.2,
                             **kwargs)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B


class MLP(nn.Module):
    def __init__(self,in_s,out_s):
        super(MLP,self).__init__()
        self.mlp=nn.Sequential( nn.Linear(int(in_s),int(in_s/2)),
                                nn.ELU(),
                                nn.Linear(int(in_s/2),int(out_s))
                                )
    def forward(self,input):
        return self.mlp(input)


class GraphGNNModelAttention(nn.Module):

    def __init__(self, c_in, c_hidden, c_out,target="higher",kl=None,skipGraph=False,
    lamb=0.1,beta=0.1,temperature=1,state_dict_weights=None,residual=True ,dropout=True,**kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        if skipGraph==True:
            c_hidden=c_in
        self.GNNlast = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden,
                            **kwargs)
        self.gnn1 = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_in,
                            **kwargs)
        self.gnn2 = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_in,
                            **kwargs)
        self.target=target
        self.lamb=lamb
        self.beta=beta
        self.dropout=dropout
        self.temperature=temperature
        self.kl=kl
        self.residual=residual
        self.classes=c_out
        self.c_hidden=c_hidden
        self.skipGraph=skipGraph


        mil2fc,mil2bag=FCLayer(c_hidden,c_out),BClassifier(c_hidden,c_out)
        self.mil2=MILNet(mil2fc,mil2bag)
        mil3fc,mil3bag=FCLayer(c_hidden,c_out),BClassifier(c_hidden,c_out)
        self.mil3=MILNet(mil3fc,mil3bag)
        millastfc,millastbag=FCLayer(c_hidden*2,c_out),BClassifier(c_hidden*2,c_out)
        self.millast=MILNet(millastfc,millastbag)



    def forward_scale(self,feats,edge_index,gnnlayer):
        r=feats
        if self.training and self.dropout:
            edge_index, _, _= dropout_node(edge_index=edge_index)
        feats=gnnlayer(feats,edge_index)# [N,250]
        if self.residual:
            feats=r+feats
        else:
            feats=feats
        return feats,edge_index

    def forward_gnn(self,x,edge_index,levels,childof,edge_index2=None,edge_index3=None):
        results={}
        featsperlevel=[]
        indecesperlevel=[]
        for i in levels.unique():
            #select indeces per level
            indeces_feats=(levels==i).nonzero().view(-1)#[N,]
            feats=x[indeces_feats]# [N,384]
            if  not self.skipGraph:
                if i==levels.min():
                    feats,_= self.forward_scale(feats,edge_index2,self.gnn1)
                else:
                    feats,_= self.forward_scale(feats,edge_index3,self.gnn2)
            featsperlevel.append(feats)
            indecesperlevel.append(indeces_feats)
        feats=torch.concat(featsperlevel) #[N+M,250]
        indeces= torch.concat(indecesperlevel).view(-1)#[N+M,]
        indeces= indeces.sort()[1]
        feats= feats[indeces]

        feats,edge_index2= self.forward_scale(feats,edge_index,self.GNNlast)
        results["childof"]=childof[indecesperlevel[1]]
        #-------------------------------------------------------------------------------
        return feats,indecesperlevel,indeces,results

    def forward_mil(self,indecesperlevel,feats,results):
        #second step: MIL
        results["lower"]=self.mil2(feats[results["childof"].unique()].view(-1,self.c_hidden))#x5
        results["higher"]=self.mil3(feats[indecesperlevel[1]].view(-1,self.c_hidden))#x20
        return results

    def forward(self, x, edge_index,levels,childof,edge_index2=None,edge_index3=None):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_index_filtered- same as edge_index without interscale connection
            levels- scale level annotation
            childof- relation child/parent per node
            batch_idx - Index of batch element for each node
        """
        #shape

        #first step: Graph processing
        feats,indecesperlevel,indeces,results=self.forward_gnn(x, edge_index, levels, childof, edge_index2, edge_index3)
        results= self.forward_mil(indecesperlevel,feats,results)

        return results

    def dismilLoss(self, criterion,y_instance_pred,prediction_bag,bag_label,lamb=1.0,beta=1.0):
        max_prediction, _ = torch.max(y_instance_pred, 0)
        bag_loss=criterion(prediction_bag.view(1,-1), bag_label.view(1,-1))
        max_loss = criterion(max_prediction.view(1,-1),bag_label.view(1,-1))
        loss = lamb*bag_loss + beta*max_loss
        return loss

    def compute_loss(self,loss_module_instance,results,bag_label):
        #predict target instance and bag loss
        y_instance_pred1,prediction_bag1,A,B=results[self.target]
        loss=self.dismilLoss(loss_module_instance,y_instance_pred1,prediction_bag1,bag_label)
        #wandb.log({"lossHigher":loss})
        if self.kl is not None:
            y_instance_pred2,prediction_bag2,A,B=results[self.kl]
            loss5=self.dismilLoss(loss_module_instance,y_instance_pred2,prediction_bag2,bag_label,lamb=(1.0-self.lamb))
            #wandb.log({"lossLower":loss5})
            loss+=loss5
            y_instance_pred3,prediction_bag3,A,B=results[self.kl]

            mse_loss=self.computeMSE(y_instance_pred3,y_instance_pred1,results["childof"])
            loss+=self.beta*mse_loss
                #wandb.log({"mse_loss": mse_loss.item()})
            kl_bag_loss= self.computeKL(prediction_bag2,prediction_bag1)
            #wandb.log({"kl_loss": self.lamb*kl_bag_loss})
            if self.lamb!=0:
                loss+= self.lamb*kl_bag_loss
        return loss
        #crf vuole un energia massima

    def computeKL(self,student,teacher):
        loss=0
        if self.classes==1:
            student= torch.sigmoid(student/self.temperature)
            teacher= torch.sigmoid(teacher / self.temperature)
            y_student= torch.concat([student,1-student],dim=1)
            y_teacher= torch.concat([teacher,1-teacher],dim=1).detach()
            loss+= F.kl_div(y_student.log(),y_teacher,reduction="batchmean",)
        else:
            for c in range(self.classes):
                y_student= torch.sigmoid(student[:,c]/self.temperature)
                y_teacher= torch.sigmoid(teacher[:,c] / self.temperature)
                y_student= torch.concat([y_student,1-y_student],dim=0)
                y_teacher= torch.concat([y_teacher,1-y_teacher],dim=0).detach()
                loss+= F.kl_div(y_student.log(),y_teacher,reduction="batchmean",)
        return loss

    def computeMSE(self,lowerscale,higherscale,childOfIndeces):
            higherscale,_=geom_nn.pool.avg_pool_x(x=higherscale,cluster=childOfIndeces,batch=childOfIndeces)
            mse_loss=F.mse_loss(lowerscale,higherscale.detach())
            return mse_loss

    def predict(self,results):
        _,prediction_bag_higher,A,B=results[self.target]
        if self.classes>1:
            higher_prediction=torch.sigmoid(prediction_bag_higher)
            if self.kl is not None:
                _,prediction_bag_lower,A,B=results[self.kl]
                lower_prediction=torch.sigmoid(prediction_bag_lower)
            else:
                lower_prediction=None
        else:
            higher_prediction=torch.sigmoid(prediction_bag_higher)[:,-1]
            if self.kl is not None:
                _,prediction_bag_lower,A,B=results[self.kl]
                lower_prediction=torch.sigmoid(prediction_bag_lower)[:,-1]
            else:
                lower_prediction=None
        return higher_prediction,lower_prediction



def selectModelMulti(model_name,num_layers,c_hidden,temperature,beta,lamb,layer_name,c_out,input_size,residual,dropout):
    if  model_name== "WithGraph_y_Higher":
        model= GraphGNNModelAttention( c_hidden=c_hidden,c_in=input_size,residual=residual,dropout=dropout, temperature=temperature,beta=beta,lamb=lamb,c_out=c_out,target="higher",kl=None,layer_name=layer_name,num_layers=num_layers,skipGraph=False,state_dict_weights=state_dict_weights)
    if model_name== "WithGraph_y_Higher_kl_Lower": 
        model= GraphGNNModelAttention(c_hidden=c_hidden,c_in=input_size,residual=residual,dropout=dropout,temperature=temperature,beta=beta,lamb=lamb,c_out=c_out,target="higher",kl="lower",layer_name=layer_name, num_layers=num_layers,skipGraph=False,state_dict_weights=state_dict_weights)
    elif model_name== "WithoutGraph_y_concat":
        model= GraphGNNModelAttention(c_hidden=c_hidden,c_in=input_size,residual=residual,dropout=dropout,temperature=temperature,beta=beta,lamb=lamb,c_out=c_out,target="concat",kl=None,layer_name=layer_name, num_layers=num_layers,skipGraph=True,state_dict_weights=state_dict_weights)
    elif model_name== "WithGraph_y_concat":
        model= GraphGNNModelAttention(c_hidden=c_hidden,c_in=input_size,residual=residual,dropout=dropout,temperature=temperature,beta=beta,lamb=lamb,c_out=c_out,target="concat",kl=None,layer_name=layer_name, num_layers=num_layers,state_dict_weights=state_dict_weights)
    elif model_name== "WithGraph_y_concat_kl_Lower":
        model= GraphGNNModelAttention(c_hidden=c_hidden,c_in=input_size,residual=residual,dropout=dropout,temperature=temperature,beta=beta,lamb=lamb,c_out=c_out,target="concat",kl="lower",layer_name=layer_name, num_layers=num_layers, state_dict_weights=state_dict_weights)
    elif model_name== "WithoutGraph":
        model= GraphGNNModelAttention(c_hidden=c_hidden,c_in=input_size,residual=residual,dropout=dropout,temperature=temperature,beta=beta,lamb=lamb,c_out=c_out,target="higher",kl=None,layer_name=layer_name,multi=True, num_layers=num_layers, dp_rate_linear=0.5,skipGraph=True,dp_rate=0.0,state_dict_weights=state_dict_weights)
    return model


