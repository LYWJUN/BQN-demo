import time
import torch
import numpy as np
from data_utils import load_data, init_stratified_dataloader
from train_test import train, val_test
from utils import hyper_para_load, count_param, fix_seed, initialize_logger
from model.BQN import Quadratic_BN
from parse import get_args


def run(args, dataset):
    dataloaders = init_stratified_dataloader(args, *dataset)
    train_loader, val_loader, test_loader = \
        dataloaders["train_dataloader"], dataloaders["val_dataloader"], dataloaders["test_dataloader"]

    (node_sz, timeseries_sz, node_feature_sz, layers, dropout,
     pooling, cluster_num) = hyper_para_load(dataset=dataset, args=args)

    # model define and load
    model = Quadratic_BN(args=args,
                         node_sz=node_sz,
                         time_series_sz=timeseries_sz,
                         corr_pearson_sz=node_feature_sz,
                         layers=layers,
                         dropout=dropout,
                         cluster_num=cluster_num,
                         pooling=pooling)
    total_parameters = count_param(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    logger = initialize_logger()

    epoch_val_roc_list, epoch_val_loss_list = [], []
    epoch_test_roc_list, epoch_test_acc_list = [], []
    epoch_test_sen_list, epoch_test_spec_list = [], []

    for epoch in range(args.epochs):
        result_train = train(model=model, optimizer=optimizer, args=args, train_loader=train_loader, epoch=epoch)
        result_val_test = val_test(model=model, args=args, val_loader=val_loader, test_loader=test_loader)

        logger.info(" | ".join([
            f'Epoch[{epoch}/{args.epochs}]',
            f'Train Loss:{result_train["train_loss"]: .3f}',
            f'Train Accuracy:{result_train["train_acc"]: .4f}',
            f'Val Loss:{result_val_test["val_loss"]:.3f}',
            f'Val Accuracy:{result_val_test["val_acc"]:.4f}',
            f'Val AUC:{result_val_test["val_roc"]:.4f}',
            f'Test Accuracy:{result_val_test["test_acc"]: .4f}',
            f'Test AUC:{result_val_test["test_roc"]:.4f}',
            f'Test Sen:{result_val_test["test_sensitivity"]:.4f}',
            f'Test Spec:{result_val_test["test_specificity"]:.4f}'
        ]))

        epoch_val_loss_list.append(result_val_test['val_loss'])
        epoch_val_roc_list.append(result_val_test['val_roc'])
        epoch_test_roc_list.append(result_val_test['test_roc'])
        epoch_test_acc_list.append(result_val_test['test_acc'])
        epoch_test_sen_list.append(result_val_test['test_sensitivity'])
        epoch_test_spec_list.append(result_val_test['test_specificity'])

    index_max = epoch_val_loss_list.index(min(epoch_val_loss_list))
    return epoch_test_acc_list[index_max], epoch_test_roc_list[index_max], epoch_test_sen_list[index_max], epoch_test_spec_list[index_max]


def main(args):
    # fix_seed(args.seed)

    # load dataset
    dataset = load_data(args)

    runs = args.runs
    run_acc_list, run_roc_list = [], []
    run_sen_list, run_spec_list = [], []
    for i in range(runs):
        print(f'run: {i} start')
        acc, roc, sen, spec = run(args, dataset)
        print(f'run: {i} is over')
        run_acc_list.append(acc)
        run_roc_list.append(roc)
        run_sen_list.append(sen)
        run_spec_list.append(spec)

    acc_mean, acc_std = np.mean(run_acc_list), np.std(run_acc_list)
    roc_mean, roc_std = np.mean(run_roc_list), np.std(run_roc_list)
    sen_mean, sen_std = np.mean(run_sen_list), np.std(run_sen_list)
    spec_mean, spec_std = np.mean(run_spec_list), np.std(run_spec_list)
    print("After ", args.runs, "runs on ", args.dataset, "!")
    print("roc_auc ± std: {:.2f}%±{:.2f}".format(roc_mean * 100, roc_std * 100),
          "mean ± std: {:.2f}%±{:.2f}".format(acc_mean * 100, acc_std * 100))
    result_file_path = args.root_path + "/result/" + args.dataset + ".csv"
    print(f"Saving results to the'{result_file_path}'")
    with open(f"{result_file_path}", 'a+') as write_obj:
        write_obj.write(f"roc:{roc_mean * 100:.2f} ± {roc_std * 100:.2f},"
                        + f"acc:{acc_mean * 100:.2f} ± {acc_std * 100:.2f},"
                        + f"sen:{sen_mean * 100:.2f} ± {sen_std * 100:.2f},"
                        + f"spec:{spec_mean * 100:.2f} ± {spec_std * 100:.2f},"
                        + f"seed:{args.seed},"
                        + f"runs:{args.runs},"
                        + f"epochs:{args.epochs},"
                        + f"batch_size:{args.batch_size},"
                        + f"base_lr:{args.base_lr},"
                        + f"target_lr:{args.target_lr},"
                        + f"wd:{args.weight_decay},"
                        + f"layers:{args.layers},"
                        + f"activation:{args.activation},"
                        + f"dropout:{args.dropout},"
                        + f"pooling:{args.pooling},"
                        + f"cluster_num:{args.cluster_num}\n"
                        )
    print()


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)