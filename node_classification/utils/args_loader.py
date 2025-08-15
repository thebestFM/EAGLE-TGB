import argparse

def set_parser():
    parser = argparse.ArgumentParser("EAGLE for Node Classification.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"],
    )
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--tppr_alpha", type=float, default=0.2)
    parser.add_argument("--tppr_beta", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--load_best_params", action="store_true")

    parser.add_argument("--gpu", type=int, default=0)

    return parser


Param_List = {
    "tgbn-trade": dict(
        topk=50, tppr_alpha=0.6, tppr_beta=0.9, gamma=0.9, window=4
    ),
    "tgbn-genre": dict(
        topk=20, tppr_alpha=0.1, tppr_beta=0.1, gamma=0.1, window=7
    ),
    "tgbn-reddit": dict(
        topk=20, tppr_alpha=0.1, tppr_beta=0.1, gamma=0.1, window=6
    ),
    "tgbn-token": dict(
        topk=20, tppr_alpha=0.1, tppr_beta=0.1, gamma=0.1, window=5
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
