import argparse

def get_args():
    parser = argparse.ArgumentParser(description='DASMIL inference')
    parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
    parser.add_argument('--modeltype', default="WithGraph_y_Higher_kl_Lower", type=str, help='train strategy')
    parser.add_argument('--n_epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--num_layers', default=1, type=int, help='number of Graph layers')
    parser.add_argument('--n_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--c_hidden', default=256, type=int, help='intermediate size ')
    parser.add_argument('--input_size', default=384, type=int, help='input size ')
    parser.add_argument('--dataset', default="cam", type=str,choices=["cam","lung","camres","lungres"], help='dataset name')
    parser.add_argument('--lamb', default=1, type=float, help='lambda')
    parser.add_argument('--residual', default=False, action="store_true")
    parser.add_argument('--beta', default=1, type=float, help='beta')
    parser.add_argument('--dropout', default=False, action="store_true")
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--scale', default="23", type=str, help='scale resolution')
    parser.add_argument('--layer_name', default="GAT", type=str, help='layer graph name')
    parser.add_argument('--temperature', default=1.5, type=float, help='temperature')
    parser.add_argument('--project', default="cam-finalv4", type=str, help='project name for wandb')
    args = parser.parse_args()
    return args