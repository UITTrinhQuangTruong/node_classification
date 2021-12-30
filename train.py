import argparse


from trainer.mlp_trainer import mlp_trainer
from trainer.sage_trainer import sage_trainer
from trainer.gat_trainer import gat_trainer


def main():
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument("--graph", type=str, default="mlp")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--save_plot", type=bool, default=False)
    parser.add_argument("--show_plot", type=bool, default=False)
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")

    parser.add_argument("--n_heads", type=int, default=3)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--attn_drop", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=0.0)

    parser.add_argument("--step-size", type=float, default=1e-3)
    parser.add_argument("-m", type=int, default=3)
    parser.add_argument("--amp", type=int, default=2)
    parser.add_argument("--vanilla", action="store_true")
    parser.add_argument(
        "--use-norm",
        action="store_true",
        help="Use symmetrically normalized adjacency matrix.",
    )
    parser.add_argument(
        "--cpu", type=bool, default=True, help="CPU mode. This option overrides --gpu."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    args = parser.parse_args()
    print(args)

    if args.graph == "mlp":
        print("Train MLP model")
        mlp_trainer(
            args.device,
            args.log_steps,
            args.num_layers,
            args.hidden_channels,
            args.dropout,
            args.lr,
            args.epochs,
            args.runs,
            args.save_model,
            args.save_plot,
            args.show_plot,
            args.path,
            args.dataset,
        )
    elif args.graph == "sage":
        print("Train Sage model")
        sage_trainer(
            args.device,
            args.log_steps,
            args.num_layers,
            args.hidden_channels,
            args.dropout,
            args.lr,
            args.epochs,
            args.runs,
            args.save_model,
            args.save_plot,
            args.show_plot,
            args.path,
            args.dataset,
        )
    elif args.graph == "gat":
        print("Train GAT model")
        gat_trainer(
            cpu=args.cpu,
            n_layers=args.num_layers,
            n_hidden=args.hidden_channels,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs,
            runs=args.runs,
            name_dataset=args.dataset,
            use_norm=args.use_norm,
            n_heads=args.n_heads,
            attn_drop=args.attn_drop,
            vanilla=args.vanilla,
            wd=args.wd,
            save_plot=args.save_plot,
            output_dir=args.path
        )
    else:
        print("Graph is wrong!!!")


if __name__ == "__main__":
    main()
