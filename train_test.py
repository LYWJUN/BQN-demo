import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import continues_mixup_data, accuracy, isfloat, optimizer_update
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report


def train(model, optimizer, args, train_loader, epoch):
    """
    model train
    """
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # train
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model = model.to(device)
    model.train()
    train_loss = 0
    train_acc_list = []
    step, total_steps = 0 + epoch * len(train_loader), len(train_loader) * args.epochs
    for time_series, node_feature, label in train_loader:
        step += 1
        time_series, node_feature, label = time_series.to(device), node_feature.to(device), label.to(device)
        time_series, node_feature, label = continues_mixup_data(
            time_series, node_feature, y=label, device=device)
        output = model(time_series, node_feature)
        label = label.float()

        optimizer_update(optimizer=optimizer, step=step, total_steps=total_steps, args=args)
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        top1 = accuracy(output, label[:, 1])[0] / 100
        train_acc_list.append(top1)

    train_loss = train_loss / (train_loader.dataset.tensors[0].shape[0] // 16)
    train_acc = np.mean(train_acc_list)

    return {"train_loss": train_loss, "train_acc": train_acc}


def val_test(model, args, val_loader, test_loader):
    """
    model validation on valid dataset and acc&roc on test dataset
    """
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # valid
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    val_loss = 0
    val_acc_list = []
    result = []
    labels = []
    for time_series, node_feature, label in val_loader:
        time_series, node_feature, label = time_series.to(device), node_feature.to(device), label.to(device)

        output = model(time_series, node_feature)
        label = label.float()

        loss = criterion(output, label)
        val_loss += loss.item()
        top1 = accuracy(output, label[:, 1])[0] / 100
        val_acc_list.append(top1)
        result += F.softmax(output, dim=1)[:, 1].tolist()
        labels += label[:, 1].tolist()

    val_loss = val_loss / ((val_loader.dataset.tensors[0].shape[0] // 16) + 1)
    val_acc = np.mean(val_acc_list)
    val_roc = roc_auc_score(labels, result)

    # test (just for experiments result show not use in def test)
    test_loss = 0
    test_acc_list = []
    result = []
    labels = []
    for time_series, node_feature, label in test_loader:
        time_series, node_feature, label = time_series.to(device), node_feature.to(device), label.to(device)

        output = model(time_series, node_feature)
        label = label.float()

        loss = criterion(output, label)
        test_loss += loss.item()
        top1 = accuracy(output, label[:, 1])[0] / 100
        test_acc_list.append(top1)
        result += F.softmax(output, dim=1)[:, 1].tolist()
        labels += label[:, 1].tolist()

    test_loss = test_loss / ((test_loader.dataset.tensors[0].shape[0] // 16) + 1)
    test_acc = np.mean(test_acc_list)
    test_roc = roc_auc_score(labels, result)

    result, labels = np.array(result), np.array(labels)
    result[result > 0.5] = 1
    result[result <= 0.5] = 0
    # metric = precision_recall_fscore_support(labels, result, average='micro')

    report = classification_report(labels, result, output_dict=True, zero_division=0)

    recall = [0, 0]
    for k in report:
        if isfloat(k):
            recall[int(float(k))] = report[k]['recall']

    return {"val_loss": val_loss, "val_acc": val_acc, "val_roc": val_roc,
            "test_loss": test_loss, "test_acc": test_acc, "test_roc": test_roc,
            "test_sensitivity": recall[-1], "test_specificity": recall[-2]}