"""Compare an embedding model to a "non-learning" frequency baseline."""
import argparse
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm

import kge.model
from kge.util.io import load_checkpoint

from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_checkpoint", help="LibKGE model checkpoint")

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


def entity_frequency(spo, direction="o"):
    """
    :param spo: torch.Tensor of triples
    :param direction: 'o' for tail entities, 's' for head entities
    :return: [(head or tail entity, frequency proportion)] for all triples
    """
    idx = 2 if direction == "o" else 0
    entities = spo[:, idx].numpy()
    counts = pd.Series(entities).value_counts(normalize=True)
    return list(zip(counts.index, counts))


def score_by_frequency(model, test_spo, direction="o"):
    """
    :param dataset: kge.dataset.Dataset
    :param test_spo: torch.Tensor of test triples
    :return: scores of test triples using an entity frequency scoring baseline
    """
    train_spo = model.dataset.split("train")
    scores = torch.zeros(
        [len(test_spo), model.dataset.num_entities()], dtype=torch.double, device="cpu",
    )

    freq = {}
    for i, triple in enumerate(test_spo):
        s, p, o = [triple[i].item() for i in range(3)]
        rel_idx = train_spo[:, 1] == p

        # get most-frequent entities for this relation in train
        if p not in freq:
            freq[p] = entity_frequency(train_spo[rel_idx], direction=direction)

        # filter out seen head/tail entities
        ent_idx = train_spo[:, 0] == s if direction == "o" else train_spo[:, 2] == o
        idx = 2 if direction == "o" else 0
        seen_entities = set(train_spo[rel_idx & ent_idx][:, idx])
        pairs = [pair for pair in freq[p] if pair[0] not in seen_entities]

        # score the remaining entities by relative frequency
        eids, eid_scores = zip(*pairs) if len(pairs) else zip(*freq[p])
        eids = torch.tensor(eids, dtype=torch.long, device="cpu")
        eid_scores = torch.tensor(eid_scores, dtype=torch.double, device="cpu")
        scores[i, eids] = eid_scores

    return scores


def score_with_model(model, test_spo, direction="o"):
    """
    :param model: kge.model.KgeModel
    :param test_spo: torch.Tensor of test triples
    :param direction: 's' for heads or 'o' for tails
    :return scores: embedding scores for predicted triples
    """
    s, p, o = [test_spo[:, i] for i in range(3)]
    if direction == "o":  # score tails
        return model.score_sp(s, p)
    return model.score_po(p, o)


def filter_false_negatives(scores, test_spo, all_spo, direction="o"):
    """
    :param scores: scores of test predictions
    :param test_spo: torch.Tensor of test triples
    :param all_spo: torch.Tensor of all triples for filtering
    :return scores: scores for each subject or object in test,
        with false negatives filtered out
    """
    for i, triple in enumerate(test_spo):
        s, p, o = [triple[i].item() for i in range(3)]

        rel_idx = all_spo[:, 1] == p
        ent_idx = (
            (all_spo[:, 0] == s) & (all_spo[:, 2] != o)
            if direction == "o"
            else (all_spo[:, 0] != s) & (all_spo[:, 2] == o)
        )
        idx = 0 if direction == "s" else 2

        false_neg_idx = all_spo[rel_idx & ent_idx][:, idx]
        scores[i, false_neg_idx] = float("-Inf")  # ranked last
    return scores


def evaluate_rankings(scores, test_spo, all_spo, direction="o", k=10):
    """
    :param scores: triple scores
    :param test_spo: torch.Tensor of test triples
    :param all_spo: torch.Tensor of all triples for filtering
    :param direction: 'o' for tail entities, 's' for head entities
    :param k: k for hits@k
    :return mrr, hits: MRR and Hits@k scores for this subset of triples
    """
    # remove true triples with same head/relation or relation/tail
    scores = filter_false_negatives(scores, test_spo, all_spo, direction=direction)
    scores[torch.isnan(scores)] = float("-Inf")

    # get the scores of the true target subjects/objects
    idx = 0 if direction == "s" else 2
    targets = test_spo[:, idx].long()
    arange = torch.arange(len(targets), dtype=torch.long, device="cpu")
    true_scores = scores[arange, targets].view(-1, 1)

    # remove the true subjects/objects from the scores so they don't factor in rankings
    scores = scores.clone()
    scores[arange, targets] = float("-Inf")

    # follow LibKGE protocol by taking the mean rank among all entities with same score
    ranks = torch.sum(scores > true_scores, dim=1, dtype=torch.double)
    num_ties = torch.sum(scores == true_scores, dim=1, dtype=torch.double)
    ranks = ranks + num_ties // 2 + 1  # ranks are one-indexed

    mrr, hits = (1 / ranks).tolist(), (ranks <= k).tolist()
    return mrr, hits


@torch.no_grad()
def main():
    args = parse_args()

    # Load model checkpoint and data
    checkpoint = load_checkpoint(args.model_checkpoint, device="cpu")
    model_pt = kge.model.KgeModel.create_from(checkpoint)
    print("Loaded model from", args.model_checkpoint)

    dataset = model_pt.dataset

    # Load all data
    train_spo, valid_spo, test_spo = [
        dataset.split(split) for split in ("train", "valid", "test")
    ]
    all_spo = torch.cat((train_spo, valid_spo, test_spo), axis=0).long()

    # Load relation ID to string mapping
    relation_ids = dataset.relation_ids()
    metric_names = ("mrr", "hits@10")
    metrics_all = defaultdict(lambda: defaultdict(list))
    dfs = []

    # Keep track of percentage of test triples per relation type
    for rid in tqdm(torch.unique(test_spo[:, 1]), desc="Relation"):
        rid = rid.item()

        # Get all test triples with this relation
        test_filt = test_spo[test_spo[:, 1] == rid]

        for direction in ["s", "o"]:  # (?, r, t) and (h, r, ?)
            metrics_mean = defaultdict(dict)

            for modelname, score_fn in zip(
                ["Model", "Baseline"], [score_with_model, score_by_frequency]
            ):

                # score test triples and evaluate rankings
                scores = score_fn(model_pt, test_filt, direction=direction)
                model_metrics = evaluate_rankings(
                    scores, test_filt, all_spo, direction=direction
                )

                for metric_name, metric in zip(metric_names, model_metrics):
                    metrics_mean[modelname][metric_name] = np.mean(metric)
                    metrics_all[modelname][metric_name].extend(metric)

            for metric_name in metric_names:
                model_metric = metrics_mean["Model"][metric_name]
                baseline_metric = metrics_mean["Baseline"][metric_name]
                diff = model_metric - baseline_metric

                line = dict(
                    relation=relation_ids[rid],
                    metric=metric_name,
                    direction=direction,
                    count=len(test_filt),
                    diff=diff,
                    model=model_metric,
                    baseline=baseline_metric,
                )

                if args.csv is not None:
                    dfs.append(pd.DataFrame.from_dict(line, orient="index").transpose())

    if args.csv is not None:
        df = pd.concat(dfs)
        df.to_csv(args.csv, index=False)
        print("Saved results to", args.csv)

    for modelname in metrics_all:
        for metric, scores in metrics_all[modelname].items():
            print(modelname, metric, np.mean(scores))


if __name__ == "__main__":
    main()
