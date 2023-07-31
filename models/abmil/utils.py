import torch.nn as nn
import torch
import torch.nn.functional as F



def init(model,state_dict_weights):
    try:
        model.load_state_dict(state_dict_weights, strict=False)
    except:
        del state_dict_weights['b_classifier.v.1.weight']
        del state_dict_weights['b_classifier.v.1.bias']
        model.load_state_dict(state_dict_weights, strict=False)
    return model


class BClassifierABMIL(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True): # K, L, N
        super(BClassifierABMIL, self).__init__()

        self.L = input_size
        self.D = 128
        self.K = 1

        self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
        self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats): # N x K, N x C
        device = feats.device
        feats = self.lin(feats)
        A_V = self.attention_V(feats) # N x V, unsorted
        A_U = self.attention_U(feats)

        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        B = torch.mm(A, feats) # compute bag representation, B in shape C x V

        C = self.classifier(B) # 1 x C x 1
        C = C.view(1, -1)
        return C,C, A, B

