import argparse


from trainer.mlp_trainer import mlp_trainer
from trainer.sage_trainer import sage_trainer


def main():
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--graph', type=str, default='mlp')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--path', type=str, default='./model.pt')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    args = parser.parse_args()
    print(args)

    if args.graph == 'mlp':
        print('Train MLP model')
        mlp_trainer(args.device,
                    args.log_steps,
                    args.num_layers,
                    args.hidden_channels,
                    args.dropout,
                    args.lr,
                    args.epochs,
                    args.runs,
                    args.save_model,
                    args.path,
                    args.dataset)
    elif args.graph == 'sage':
        print('Train Sage model')
        sage_trainer(args.device,
                     args.log_steps,
                     args.num_layers,
                     args.hidden_channels,
                     args.dropout,
                     args.lr,
                     args.epochs,
                     args.runs,
                     args.save_model,
                     args.path,
                     args.dataset)

    else:
        print('Graph is wrong!!!')


if __name__ == '__main__':
    main()
