import argparse

def set_parser():
    parser = argparse.ArgumentParser("EAGLE for Link Prediction.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-comment", "tgbl-flight"],
    )

    parser.add_argument("--num_runs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--model", type=str, default="hybrid", choices=["structure", "time", "hybrid"])
    parser.add_argument("--topk_struct", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--topk_time", type=int, default=15)
    parser.add_argument("--time_sample_strategy", type=str, default="last")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_dims", type=int, default=100)
    parser.add_argument("--yita", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--load_best_params", action="store_true")

    parser.add_argument("--gpu", type=int, default=0)

    return parser


Param_List = {
    "tgbl-wiki": dict(
        model="hybrid", alpha=0.88, beta=0.75, yita=0.0003
    ),
    "tgbl-review": dict(
        model="time", topk_time=20, lr=0.005, weight_decay=0.0, num_layers=2
    ),
    "tgbl-coin": dict(
        model="hybrid", alpha=0.2, beta=0.8, weight_decay=0.0, num_layers=3, yita=0.1, seed=2025
    ),
    "tgbl-comment": dict(
        model="time", lr=0.005, weight_decay=0.0, patience=4
    ),
    "tgbl-flight": dict(
        model="time", patience=4
    ),
}


def load_args():
    parser = set_parser()
    args = parser.parse_args()

    if args.load_best_params:
        overrides = Param_List.get(args.dataset_name, {})
        for key, value in overrides.items():
            setattr(args, key, value)

    return args
