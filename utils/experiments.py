import itertools
import copy


def launch_main(args):
    args1 = copy.copy(args)
    args1.modeltype = "WithGraph_y_Higher_kl_Lower"
    args1.lamb = 1
    args1.beta = 1
    args1.temperature = 1.5
    args1.scale = "23"
    args1.seed = 42
    args1.lr = 0.0002
    args1.weight_decay = 0.005
    args1.c_hidden = 384
    args1.layer_name = "GAT"
    args1.residual = True
    args1.project = "mainlonger"
    return [args1]
