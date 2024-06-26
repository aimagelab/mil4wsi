import sys
import vision_transformer as vits
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
import torch
from models.utils.basemodel import Baseline
from utilsmil4wsi.utils2 import dropout_node




class HIPT_None_FC(Baseline):
    def __init__(self,args,state_dict_weights, path_input_dim=384, size_arg = "small", dropout=0.25, n_classes=1):
        super(HIPT_None_FC, self).__init__(args,state_dict_weights)
        self.size_dict_path = {"small": [path_input_dim, 256, 256], "big": [path_input_dim, 512, 384]}
        size = self.size_dict_path[size_arg]

        ### Local Aggregation
        self.local_phi = nn.Sequential(
            nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25),
        )
        self.local_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)

        ### Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25),
        )
        self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
        self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])
        self.classifier = nn.Linear(size[1], n_classes)


    def forward(self, h, edge_index,level,childof,edge_index2,edge_index3, **kwargs):
        results={}
        x_256 = h

        ### Local
        h_256 = self.local_phi(x_256)
        A_256, h_256 = self.local_attn_pool(h_256)
        #A_256 = A_256.squeeze(dim=2) # A = torch.transpose(A, 1, 0)
        A_256 = F.softmax(A_256, dim=1)
        h_4096 = torch.bmm(torch.transpose(A_256[None,:,:],1,2), h_256[None,:,:]).squeeze(dim=1)

        ### Global

        A_4096, h_4096 = self.global_attn_pool(h_4096)
        A_4096 = torch.transpose(A_4096, 1, 0)
        A_4096 = F.softmax(A_4096, dim=1)
        h_path = torch.mm(A_4096, h_4096)
        h_path = self.global_rho(h_path)
        logits = self.classifier(h_path)

        #Y_hat = torch.topk(logits, 1, dim = 1)[1]
        #Y_prob = F.softmax(logits, dim = 1)
        results["higher"]= logits,logits, A_4096,h_path
        return results#logits, Y_prob, Y_hat, None, None



######################################
# 3-Stage HIPT Implementation (With Local-Global Pretraining) #
######################################
class HIPT_LGP_FC(Baseline):
    def __init__(self,args,state_dict_weights, path_input_dim=384,  size_arg = "small", dropout=0.25, n_classes=1,
     pretrain_4k='None', freeze_4k=False, pretrain_WSI='None', freeze_WSI=False):
        super(HIPT_LGP_FC, self).__init__(args,state_dict_weights)
        self.size_dict_path = {"small": [384, 28, 28], "big": [1024, 512, 384]}
        #self.fusion = fusion
        size = self.size_dict_path[size_arg]

        ### Local Aggregation
        self.local_vit = vits.__dict__["vit_small"](patch_size=384, num_classes=0)
        if pretrain_4k != 'None':
            print("Loading Pretrained Local VIT model...",)
            state_dict = torch.load('../../HIPT_4K/Checkpoints/%s.pth' % pretrain_4k, map_location='cpu')['teacher']
            state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
            state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.local_vit.load_state_dict(state_dict, strict=False)
            print("Done!")
        if freeze_4k:
            print("Freezing Pretrained Local VIT model")
            for param in self.local_vit.parameters():
                param.requires_grad = False
            print("Done")

        ### Global Aggregation
        self.pretrain_WSI = pretrain_WSI
        if pretrain_WSI != 'None':
            pass
        else:
            self.global_phi = nn.Sequential(nn.Linear(384, 28), nn.ReLU(), nn.Dropout(0.25))
            self.global_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=28, nhead=1, dim_feedforward=28, dropout=0.25, activation='relu'
                ),
                num_layers=1
            )
            self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
            self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])

        self.classifier = nn.Linear(size[1], n_classes)


    def forward(self, h_4096, edge_index,level,childof,edge_index2,edge_index3,**kwargs):
        ### Local
        #h_4096 = self.local_vit(x_256.unfold(1, 16, 16).transpose(1,2))
        #if self.training and self.args.dropout:
        edge_index, edge_mask, edge_node= dropout_node(edge_index=edge_index,p=0.5)
        h_4096=h_4096[edge_node]
        results={}
        ### Global
        if self.pretrain_WSI != 'None':
            h_WSI = self.global_vit(h_4096.unsqueeze(dim=0))
        else:
            h_4096 = self.global_phi(h_4096)
            h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
            A_4096, h_4096 = self.global_attn_pool(h_4096)
            A_4096 = torch.transpose(A_4096, 1, 0)
            A_4096 = F.softmax(A_4096, dim=1)
            h_path = torch.mm(A_4096, h_4096)
            h_WSI = self.global_rho(h_path)

        logits = self.classifier(h_WSI)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        results["higher"]= logits,logits, A_4096,h_path
        return results


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.local_vit = nn.DataParallel(self.local_vit, device_ids=device_ids).to('cuda:0')
            if self.pretrain_WSI != 'None':
                self.global_vit = nn.DataParallel(self.global_vit, device_ids=device_ids).to('cuda:0')

        if self.pretrain_WSI == 'None':
            self.global_phi = self.global_phi.to(device)
            self.global_transformer = self.global_transformer.to(device)
            self.global_attn_pool = self.global_attn_pool.to(device)
            self.global_rho = self.global_rho.to(device)

        self.classifier = self.classifier.to(device)

