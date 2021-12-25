import os

import torch
from torchsummary import summary

from model.architectures import MLP
from model.loss import nll_loss
from model.metric import accuracy_mlp
from logs.logger import Logger
from tools.data import load_data
from model.metric import Evaluator


def train(model, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def mlp_trainer(device=0,
                log_steps=1,
                num_layers=3,
                hidden_channels=256,
                dropout=0.5,
                lr=0.01,
                epochs=500,
                runs=10,
                save_model=False,
                save_plot=False,
                show_plot=False,
                output_dir='.',
                name_dataset='ogbn-arxiv'):

    device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if save_model:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(
            output_dir, f'mlp_{name_dataset}_r{runs}_e{epochs}_n{num_layers}.pt')

    dataset = load_data(name=name_dataset, transform=False)
    data = dataset[0]

    if name_dataset == 'cora':
        split_idx = {'train': data.train_mask.nonzero().reshape(-1),
                     'test': data.test_mask.nonzero().reshape(-1),
                     'valid': data.val_mask.nonzero().reshape(-1)}
        data.y = data.y.reshape(-1, 1)
    else:
        split_idx = dataset.get_idx_split()

    x = data.x
    x = x.to(device)

    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    model = MLP(x.size(-1), hidden_channels, dataset.num_classes,
                num_layers, dropout).to(device)
    # summary model
    summary(model, (hidden_channels, x.size(-1)))

    evaluator = Evaluator()
    logger = Logger(runs)

    best_valid_acc = 0
    for run in range(runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, 1 + epochs):
            loss = train(model, x, y_true, train_idx, optimizer)
            result = accuracy_mlp(model, x, y_true, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            logger.add_result(run, result)

            if epoch % log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

            # best_test_acc = test_acc if test_acc > best_test_acc else best_test_acc
            if save_model:
                if valid_acc > best_valid_acc:
                    print(f'Save model in {output_path}')
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), output_path)

        logger.print_statistics(run)
    logger.print_statistics()
    logger.visualize(save_plot=save_plot,
                     output_dir=output_dir, show_plot=show_plot)
