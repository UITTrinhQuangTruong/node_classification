import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from model.architectures import MLP
from model.loss import nll_loss
from model.metric import accuracy_mlp


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
                output_path='./model.pt',
                name_dataset='ogbn-arxiv'):

    device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name=name_dataset)
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    x = data.x
    x = x.to(device)

    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    model = MLP(x.size(-1), hidden_channels, dataset.num_classes,
                num_layers, dropout).to(device)
    print(x.size(-1), hidden_channels, dataset.num_classes, num_layers, dropout)
    evaluator = Evaluator(name=name_dataset)

    best_valid_acc = 0
    for run in range(runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, 1 + epochs):
            loss = train(model, x, y_true, train_idx, optimizer)
            result = accuracy_mlp(model, x, y_true, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result

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
