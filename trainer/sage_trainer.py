import os

import torch
from model.metric import Evaluator


from tools.data import load_data
from model.architectures import SAGE, SAGE_norm
from model.loss import nll_loss
from model.metric import accuracy_sage
from logs.logger import Logger


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def sage_trainer(device=0,
                 log_steps=1,
                 num_layers=3,
                 hidden_channels=256,
                 dropout=0.5,
                 lr=0.01,
                 epochs=300,
                 runs=10,
                 save_model=False,
                 output_dir='.',
                 name_dataset='ogbn-arxiv'):

    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if save_model:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(
            output_dir, f'sage_{name_dataset}_r{runs}_e{epochs}_n{num_layers}.pt')

    dataset = load_data(name_dataset, transform=True)
    data = dataset[0]
    if name_dataset == 'cora':
        split_idx = {'train': data.train_mask.nonzero().reshape(-1),
                     'test': data.test_mask.nonzero().reshape(-1),
                     'valid': data.val_mask.nonzero().reshape(-1)}
        data.y = data.y.reshape(-1, 1)
    else:
        split_idx = dataset.get_idx_split()

    train_idx = split_idx['train'].to(device)
    if name_dataset == 'ogbn-arxiv' or name_dataset == 'cora':
        data.adj_t = data.adj_t.to_symmetric()

        model = SAGE_norm(data.num_features, hidden_channels,
                          dataset.num_classes, num_layers,
                          dropout).to(device)

    else:
        model = SAGE(data.num_features, hidden_channels,
                     dataset.num_classes, num_layers,
                     dropout).to(device)

    data = data.to(device)

    # summary model
    # summary(model, (hidden_channels, data.num_features))
    print(model)

    evaluator = Evaluator()
    logger = Logger(runs)

    best_valid_acc = 0
    for run in range(runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = train(model, data, train_idx, optimizer)
            result = accuracy_sage(model, data, split_idx, evaluator)
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result

            if epoch % log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

            if save_model:
                if valid_acc > best_valid_acc:
                    print(f'Save model in {output_path}')
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), output_path)

        logger.print_statistics(run)
    logger.print_statistics()
