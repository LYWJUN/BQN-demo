import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict, Any
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from utils import StandardScaler_crossROI
from nilearn import connectome


# data_utils: Processing of data
def load_data(args):
    if args.dataset == 'ABIDE':
        data_path = args.data_dir + '/' + args.dataset.lower() + '.npy'
        data = np.load(data_path, allow_pickle=True).item()
        data_timeseries = data['timeseires']  # [1009, 200, 100]
        data_label = data['label']  # [1009,], 0:control, 1:patient
        data_pearson = data['corr']  # [1009, 200, 200]
        site = data['site']

        data_timeseries = StandardScaler_crossROI(data_timeseries)

        (data_timeseries, data_label, data_pearson) = \
            [torch.from_numpy(data).float() for data in (data_timeseries, data_label, data_pearson)]

        return data_timeseries, data_pearson, data_label, site


def init_stratified_dataloader(args,
                               final_timeseries: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array) -> dict[str, Any]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseries.shape[0]
    train_length = int(length * args.Train_prop)
    val_length = int(length * args.Val_prop)
    test_length = length - train_length - val_length

    spilt1 = StratifiedShuffleSplit(n_splits=1, train_size=train_length, test_size=length-train_length, random_state=args.seed)
    for train_index, val_and_test_index in spilt1.split(final_timeseries, stratified):
        final_timeseries_train, final_pearson_train, labels_train = final_timeseries[
            train_index], final_pearson[train_index], labels[train_index]
        final_timeseries_val_and_test, final_pearson_val_and_test, labels_val_and_test = final_timeseries[
            val_and_test_index], final_pearson[val_and_test_index], labels[val_and_test_index]
        stratified = stratified[val_and_test_index]

    spilt2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length)
    for val_index, test_index in spilt2.split(final_timeseries_val_and_test, stratified):
        final_timeseries_val, final_pearson_val, labels_val = final_timeseries_val_and_test[
            val_index], final_pearson_val_and_test[val_index], labels_val_and_test[val_index]
        final_timeseries_test, final_pearson_test, labels_test = final_timeseries_val_and_test[
            test_index], final_pearson_val_and_test[test_index], labels_val_and_test[test_index]

    train_dataset = torch.utils.data.TensorDataset(final_timeseries_train, final_pearson_train, labels_train)
    val_dataset = torch.utils.data.TensorDataset(final_timeseries_val, final_pearson_val, labels_val)
    test_dataset = torch.utils.data.TensorDataset(final_timeseries_test, final_pearson_test, labels_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    return {"train_dataloader": train_dataloader, "val_dataloader": val_dataloader, "test_dataloader": test_dataloader}

