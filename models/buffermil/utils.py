import torch.nn as nn
import torch
import torch.nn.functional as F


class BClassifierBuffer(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True): # K, L, N
        super(BClassifierBuffer, self).__init__()
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
        self.sigma=0.05

    def forward(self, feats_slide, feats_buffer,use_buffer,aggregationtype: None): # N x K, M x K
        device = feats_slide.device
        feats_slide = self.lin(feats_slide)
        V = self.v(feats_slide) # N x V
        Q = self.q(feats_slide).view(feats_slide.shape[0], -1) # N x Q

        q_max = self.q(feats_buffer) # MxQ
        A = torch.mm(Q, q_max.transpose(0, 1)) #NxM
        if use_buffer:
            if aggregationtype== "mean":
                A=A.mean(dim=1)[:,None]     #N,1
            elif aggregationtype=="max":
                A=A.max(dim=1)[0][:,None]   #N,1

        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x 1,
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape 1 x V

        B = B.view(1, B.shape[0], B.shape[1]) # 1 x 1 x V
        C = self.fcc(B) # 1 x 1 x 1
        C = C.view(1, -1)
        return C, A, B


class BClassifierNew(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True): # K, L, N
        super(BClassifierNew, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(nn.Dropout(dropout_v),nn.Linear(input_size, input_size))

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats,concept_feat): # N x K, N x C
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted

        q_max = self.q(concept_feat) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(-1)
        return C, A, B


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class MILNetBuffer(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNetBuffer, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.buffer=None

    def forward(self, x,inference):
        #if self.buffer is None:
        feats, classes = self.i_classifier(x)
        # handle multiple classes without for loop
        _, m_indices = torch.sort(classes, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K
        prediction_bag, A, B = self.b_classifier(feats, m_feats,inference,None)
        #else:
        #    for m_feats in self.buffer:
        #        print("error")

        return classes, prediction_bag, A, B

    def store(self,feats):
        if self.buffer is None:
            self.buffer=feats
        else:
            self.buffer= torch.concatenate([self.buffer,feats])

    def bufferinference(self,feats,inference,aggregationtype):
        prediction_bag, A, B = self.b_classifier(feats, self.buffer,inference,aggregationtype)
        #else:
        #    for m_feats in self.buffer:
        #        print("error")

        return prediction_bag, prediction_bag, A, B



def init(model,state_dict_weights):
    try:
        model.load_state_dict(state_dict_weights, strict=False)
    except:
        del state_dict_weights['b_classifier.v.1.weight']
        del state_dict_weights['b_classifier.v.1.bias']
        model.load_state_dict(state_dict_weights, strict=False)
    return model

class MLP(nn.Module):
    def __init__(self,in_s,out_s):
        super(MLP,self).__init__()
        self.mlp=nn.Sequential( nn.Linear(int(in_s),int(in_s/2)),
                                nn.ELU(),
                                nn.Linear(int(in_s/2),int(out_s))
                                )
    def forward(self,input):
        return self.mlp(input)
