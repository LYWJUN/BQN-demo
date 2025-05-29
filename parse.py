import argparse


def get_args():
    """ create parser """
    parser = argparse.ArgumentParser(description='BQN hyper parameters')
    parser.add_argument('--device', type=int, default=0, help="cuda:0")
    parser.add_argument('--root_path', type=str, default="/home/ywl/GNN-codes/BrainNetwork/BQN_Demo")
    parser.add_argument('--data_dir', type=str, default="/home/ywl/Load_datasets/Brain_data/FMRI")

    parser.add_argument('--dataset', default='ABIDE', help='brain dataset',choices=['ABIDE', 'ADNI', 'ADHD', 'PPMI'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', default=5, help='repeat time')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--Train_prop', default=0.7)
    parser.add_argument('--Val_prop', default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--target_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='leaky_relu')   # gelu, leaky_relu, elu, sigmoid
    parser.add_argument('--pooling', type=bool, default=True)
    parser.add_argument('--cluster_num', type=int, default=4)

    """ The command line reads the parameters """
    return parser.parse_args()
