import argparse
import math
import time
from torch_geometric.datasets import Planetoid
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from tools.data import load_data
from model.architectures import *
from model.metric import Evaluator
from ogb.nodeproppred import DglNodePropPredDataset, PygNodePropPredDataset
import torch_geometric.transforms as T

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


def gen_model(
    use_labels=True,
    n_hidden=3,
    use_norm=False,
    n_layers=3,
    n_heads=3,
    dropout=0.75,
    attn_drop=0.05,
):
    norm = "both" if use_norm else "none"

    if use_labels:
        model = GAT(
            in_feats + n_classes,
            n_classes,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            activation=F.relu,
            dropout=dropout,
            attn_drop=attn_drop,
            norm=norm,
        )
    else:
        model = GAT(
            in_feats,
            n_classes,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            activation=F.relu,
            dropout=dropout,
            attn_drop=attn_drop,
            norm=norm,
        )

    return model


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator):
    return evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train_vanilla(
    model,
    graph,
    labels,
    train_idx,
    optimizer,
    use_labels=True,
):
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    loss.backward()
    optimizer.step()

    return loss, pred


def train(
    model,
    graph,
    labels,
    train_idx,
    optimizer,
    use_labels=True,
    m=3,
    step_size=1e-3,
    amp=2,
):
    model.train()
    optimizer.zero_grad()

    feat = graph.ndata["feat"].to(device)
    perturb = th.FloatTensor(*feat.shape).uniform_(-step_size, step_size).to(device)

    unlabel_idx = list(set(range(perturb.shape[0])) - set(train_idx))
    perturb.data[unlabel_idx] *= amp

    perturb.requires_grad_()
    feat_input = feat + perturb

    mask_rate = 0.5
    mask = th.rand(train_idx.shape) < mask_rate
    train_labels_idx = train_idx[mask]
    train_pred_idx = train_idx[~mask]

    feat_input = add_labels(feat_input, labels, train_labels_idx)
    pred = model(graph, feat_input)
    loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    loss /= m

    for _ in range(m - 1):
        loss.backward()
        perturb_data = perturb[train_idx].detach() + step_size * th.sign(
            perturb.grad[train_idx].detach()
        )
        perturb.data[train_idx] = perturb_data.data
        perturb_data = perturb[unlabel_idx].detach() + amp * step_size * th.sign(
            perturb.grad[unlabel_idx].detach()
        )
        perturb.data[unlabel_idx] = perturb_data.data
        perturb.grad[:] = 0

        feat_input = feat + perturb
        feat_input = add_labels(feat_input, labels, train_labels_idx)
        pred = model(graph, feat_input)
        loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
        loss /= m

    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )


def run(
    graph,
    labels,
    train_idx,
    val_idx,
    test_idx,
    evaluator,
    lr=0.002,
    vanilla=True,
    epochs=100,
    use_labels=False,
    wd=5e-4,
    n_running=1,
    n_hidden=3,
    use_norm=False,
    n_layers=3,
    n_heads=3,
    dropout=0.75,
    attn_drop=0.05,
):
    # define model and optimizer
    model = gen_model(
        n_hidden=n_hidden,
        use_norm=use_norm,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        attn_drop=attn_drop,
    )
    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, epochs + 1):
        tic = time.time()
        adjust_learning_rate(optimizer, lr, epoch)

        if vanilla:
            f = train_vanilla
        else:
            f = train
        loss, pred = f(
            model,
            graph,
            labels,
            train_idx,
            optimizer,
            use_labels,
        )
        acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
            model,
            graph,
            labels,
            train_idx,
            val_idx,
            test_idx,
            use_labels,
            evaluator,
        )

        toc = time.time()
        total_time += toc - tic

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc

    print("*" * 50)
    print(f"Average epoch time: {total_time / epochs}, Test acc: {best_test_acc}")

    return best_val_acc, best_test_acc


def count_parameters():
    model = gen_model()
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def gat_trainer(
    cpu=True,
    gpu=0,
    name_dataset="ogbn-arxiv",
    runs=10,
    lr=0.002,
    use_labels=True,
    vanilla=True,
    wd=5e-4,
    epochs=100,
    n_hidden=3,
    use_norm=False,
    n_layers=3,
    n_heads=3,
    dropout=0.75,
    attn_drop=0.05,
):
    global device, in_feats, n_classes, epsilon
    if cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % gpu)

    # load data
    if name_dataset == "ogbn-arxiv":
        data = DglNodePropPredDataset(name=name_dataset)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        graph, labels = data[0]

    elif name_dataset == "cora":
        dataset = Planetoid("Planetoid", name="Cora", transform=T.ToSparseTensor())
        data = dataset[0]
        split_idx = {
            "train": data.train_mask.nonzero().reshape(-1),
            "test": data.test_mask.nonzero().reshape(-1),
            "valid": data.val_mask.nonzero().reshape(-1),
        }
        data.y = data.y.reshape(-1, 1)
        train_idx = split_idx["train"].to(device)
        graph, labels = data

    evaluator = Evaluator()
    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    # graph.create_format_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []

    for i in range(1, runs + 1):
        print(f"Run time = {i}")
        val_acc, test_acc = run(
            graph=graph,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            evaluator=evaluator,
            n_running=i,
            use_labels=use_labels,
            lr=lr,
            wd=wd,
            vanilla=vanilla,
            epochs=epochs,
            n_hidden=n_hidden,
            use_norm=use_norm,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            attn_drop=attn_drop,
        )
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        print(f"Val_acc = {val_acc}, Test_acc = {test_acc}")

    print(f"Runned {runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters()}")
