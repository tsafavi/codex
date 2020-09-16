import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "size", choices=["s", "m", "l"], help="CoDEx dataset to download model(s)"
    )

    parser.add_argument(
        "task",
        choices=["triple-classification", "link-prediction"],
        help="Task to download model(s) for",
    )

    parser.add_argument(
        "models",
        nargs="+",
        choices=["rescal", "transe", "complex", "conve", "tucker"],
        help="Model(s) to download for this task",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv("pretrained.csv")
    df = df[
        (df["size"] == args.size)
        & (df["task"] == args.task)
        & (df["model"].isin(args.models))
    ]

    for model, link in zip(df["model"], df["link"]):
        dst = os.path.join(
            "models", args.task, "codex-" + args.size, model, "checkpoint_best.pt"
        )
        if not os.path.exists(dst):
            os.system("curl -L {} -o {}".format(link, dst))
            print("Downloaded LibKGE checkpoint to", dst)
        else:
            print("Skipping download of", dst, "because path already exists")
