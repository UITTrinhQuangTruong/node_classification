import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from model.architectures import SAGE
from model.loss import nll_loss
from model.metric import accuracy_sage


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
                 output_path='./model.pt',
                 name_dataset='ogbn-arxiv'):

    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products',
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = SAGE(data.num_features, hidden_channels,
                 dataset.num_classes, num_layers,
                 dropout).to(device)

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-products')

    best_valid_acc = 0
    for run in range(runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = train(model, data, train_idx, optimizer)
            result = accuracy_sage(model, data, split_idx, evaluator)

            if epoch % log_steps == 0:
                train_acc, valid_acc, test_acc = result
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