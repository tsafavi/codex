import argparse
import os
import torch

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from kge import Config
import kge.model
import kge.util.sampler
from kge.util.io import load_checkpoint

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_files", nargs="+", help="LibKGE model checkpoint(s)")

    parser.add_argument(
        "--size",
        default="s",
        choices=["s", "m"],
        help=("CoDEx dataset size, for --negative=codex only"),
    )

    parser.add_argument(
        "--negative",
        default="codex",
        choices=["frequency", "uniform", "codex"],
        help=("Type of negative sampling to use. Default is CoDEx hard negatives"),
    )

    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "CSV filename to save results. "
            "Default None; if an argument is provided, "
            "writes results to the specified file."
        ),
    )

    return parser.parse_args()


def get_X_y(model, pos_spo, neg_spo):
    """
    :param model: kge.model.KgeModel
    :param pos_spo: torch.Tensor of positive triples
    :param neg_spo: torch.Tensor of negative triples
    :return X: torch.Tensor of [pos_scores, neg_scores]
    :return y: torch.Tensor of [1s, 0s]
    """
    pos_scores = model.score_spo(*[pos_spo[:, i] for i in range(3)], direction="o")
    neg_scores = model.score_spo(*[neg_spo[:, i] for i in range(3)], direction="o")

    X = torch.reshape(torch.cat((pos_scores, neg_scores)), (-1, 1))
    y = torch.cat(
        (
            torch.ones_like(pos_scores, device="cpu"),
            torch.zeros_like(neg_scores, device="cpu"),
        )
    )

    return X, y


def generate_neg_spo(dataset, split, negative_type="uniform", num_samples=1):
    """
    :param dataset: kge.dataset.Dataset
    :param split: one of "valid", "test"
    :param negative_type: one of "uniform", "frequency"
    :param num_samples: number of negatives per positive
    :return: torch.Tensor of randomly generated negative triples
    """
    # Sample corrupted object entities
    if negative_type == "uniform":
        sampler = kge.util.sampler.KgeUniformSampler(
            Config(), "negative_sampling", dataset
        )
    elif negative_type == "frequency":
        sampler = kge.util.sampler.KgeFrequencySampler(
            Config(), "negative_sampling", dataset
        )
    else:
        raise ValueError(f"Negative sampling type {negative_type} not recognized")

    print(
        "Generating",
        num_samples,
        "negatives per positive with",
        negative_type,
        "sampling on the",
        split,
        "split",
    )

    spo = dataset.split(split)
    neg_o = sampler.sample(spo, 2, num_samples=num_samples)

    neg_spo = torch.cat(
        (
            torch.repeat_interleave(spo[:, :2].long(), num_samples, dim=0),
            torch.reshape(neg_o, (-1, 1)),
        ),
        dim=1,
    )

    return neg_spo


def load_neg_spo(dataset, size="s"):
    """
    :param dataset: kge.dataset.Dataset
    :return: torch.Tensor of negative triples loaded from directory
    """
    negs = []

    for split in ("valid_negatives", "test_negatives"):
        triples = pd.read_csv(
            os.path.join("data/triples/codex-" + size, split + ".txt"),
            sep="\t",
            header=None,
        ).values

        # Convert string IDs to integer IDs
        entity_ids = dict(map(reversed, enumerate(dataset.entity_ids())))
        relation_ids = dict(map(reversed, enumerate(dataset.relation_ids())))

        triples = [
            [entity_ids[s], relation_ids[p], entity_ids[o]] for (s, p, o) in triples
        ]

        negs.append(torch.tensor(triples, dtype=torch.long, device="cpu"))

    return negs


def get_threshold(scores, labels):
    """
    :param scores: torch.tensor of prediction scores
    :param labels: torch.tensor of triple labels
    :return threshold: best decision threshold for these scores
    """
    predictions = ((scores.view(-1, 1) >= scores.view(1, -1)).long()).t()

    accuracies = (predictions == labels.view(-1)).float().sum(dim=1)
    accuracies_max = accuracies.max()

    threshold = scores[accuracies_max == accuracies].min().item()
    return threshold


@torch.no_grad()
def main():
    args = parse_args()

    # Load first model, get dataset
    # Assumes all models trained on same data
    checkpoint = load_checkpoint(args.model_files[0], device="cpu")
    model = kge.model.KgeModel.create_from(checkpoint)
    dataset = model.dataset

    splits = ("valid", "test")
    valid_spo, test_spo = [dataset.split(split).long() for split in splits]

    if args.negative in ("uniform", "frequency"):
        valid_neg_spo, test_neg_spo = [
            generate_neg_spo(dataset, split, negative_type=args.negative)
            for split in splits
        ]
    else:
        valid_neg_spo, test_neg_spo = load_neg_spo(dataset, size=args.size)
        print(
            f"Loaded {len(valid_neg_spo)} valid negatives",
            f"and {len(test_neg_spo)} test negatives",
        )

    valid_spo_all = torch.cat((valid_spo, valid_neg_spo))
    test_spo_all = torch.cat((test_spo, test_neg_spo))

    metrics = []
    dfs = []

    for model_file in args.model_files:
        if os.path.exists(model_file):
            checkpoint = load_checkpoint(model_file, device="cpu")
            model = kge.model.KgeModel.create_from(checkpoint)

            # Score negative and positive validation triples
            X_valid, y_valid = get_X_y(model, valid_spo, valid_neg_spo)
            X_test, y_test = get_X_y(model, test_spo, test_neg_spo)

            valid_relations = valid_spo_all[:, 1].unique()
            test_relations = test_spo_all[:, 1].unique()

            y_pred_valid = torch.zeros(y_valid.shape, dtype=torch.long, device="cpu")
            y_pred_test = torch.zeros(y_test.shape, dtype=torch.long, device="cpu")

            ############################################################################
            # begin credits to https://github.com/uma-pi1/kge/blob/triple_classification/kge/job/triple_classification.py#L302 #
            ############################################################################
            REL_KEY = -1
            thresholds = {r: -float("inf") for r in range(dataset.num_relations())}
            thresholds[REL_KEY] = -float("inf")

            for r in valid_relations:  # set a threshold for each relation
                current_rel = valid_spo_all[:, 1] == r
                threshold = get_threshold(X_valid[current_rel], y_valid[current_rel])
                thresholds[r.item()] = threshold

                predictions = X_valid[current_rel] >= threshold
                y_pred_valid[current_rel] = predictions.view(-1).long()

            # also set a global threshold for relations unseen in valid set
            thresholds[REL_KEY] = get_threshold(X_valid, y_valid)

            for r in test_relations:  # get predictions based on validation thresholds
                key = r.item() if r.item() in thresholds else REL_KEY
                threshold = thresholds[key]

                current_rel = test_spo_all[:, 1] == r
                predictions = X_test[current_rel] >= threshold

                y_pred_test[current_rel] = predictions.view(-1).long()
            ############################################################################
            #                                end credits                               #
            ############################################################################

            y_test = y_test.numpy()
            y_pred_test = y_pred_test.numpy()

            line = dict(
                valid_accuracy=accuracy_score(y_valid, y_pred_valid),
                valid_f1=f1_score(y_valid, y_pred_valid),
                test_accuracy=accuracy_score(y_test, y_pred_test),
                test_f1=f1_score(y_test, y_pred_test),
                model_file=model_file,
            )

            metrics.append(line)

            if args.csv is not None:
                dfs.append(pd.DataFrame.from_dict(line, orient="index").transpose())

    if args.csv is not None:
        df = pd.concat(dfs)
        df.to_csv(args.csv, index=False)
        print("Saved results to", args.csv)

    for metric in metrics:
        for key, val in metric.items():
            print(f"{key}: {val}")
        print()


if __name__ == "__main__":
    main()
