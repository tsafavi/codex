"""
Triple classification using different
negative generation strategies and calibration techniques
"""
import argparse
import os
import torch

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

import kge.model
import kge.config
import kge.util.sampler
from kge.util.io import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_files', nargs='+',
        help='LibKGE model checkpoint(s)'
    )

    parser.add_argument(
        '--size', default='s', choices=['s', 'm'],
        help=(
            'CoDEx dataset size, '
            'for --negative=codex only'
        )
    )

    parser.add_argument(
        '--negative', default='codex',
        nargs='+',
        choices=['frequency', 'uniform', 'codex'],
        help=(
            'Types of negative sampling to use. '
            'Default is CoDEx hard negatives'
        )
    )

    parser.add_argument(
        '--calib-type', default='isotonic',
        choices=['sigmoid', 'isotonic'],
        help=(
            'Calibrator type. '
            'Default is sigmoid (Platt scaling)'
        )
    )

    return parser.parse_args()


def get_X_y(model, pos_spo, neg_spo):
    """
    :param model: kge.model.KgeModel
    :param pos_spo: torch.Tensor of positive triples
    :param neg_spo: torch.Tensor of negative triples
    :return X: torch.Tensor of [pos_scores, neg_scores]
    :return y: torchh.Tensor of [1s, 0s]
    """
    pos_scores = model.score_spo(
        *[pos_spo[:, i] for i in range(3)], direction='o')
    neg_scores = model.score_spo(
        *[neg_spo[:, i] for i in range(3)], direction='o')

    X = torch.reshape(
        torch.cat((pos_scores, neg_scores)), (-1, 1))
    y = torch.cat(
        (torch.ones_like(pos_scores),
         torch.zeros_like(neg_scores)))

    return X, y


def generate_neg_spo(
        dataset, split,
        negative_type='uniform', num_samples=1):
    """
    :param dataset: kge.dataset.Dataset
    :param split: one of 'valid', 'test'
    :param negative_type: one of 'uniform', 'frequency'
    :param num_samples: number of negatives per positive
    :return: torch.Tensor of randomly generated negative triples
    """
    # Sample corrupted object entities
    config = kge.config.Config()

    if negative_type == 'uniform':
        sampler = kge.util.sampler.KgeUniformSampler(
            config, 'negative_sampling', dataset)
    elif negative_type == 'frequency':
        sampler = kge.util.sampler.KgeFrequencySampler(
            config, 'negative_sampling', dataset)

    print('Generating', num_samples, 'negatives per positive with',
          negative_type, 'sampling on the', split, 'split')

    spo = dataset.split(split)
    neg_o = sampler.sample(spo, 2, num_samples=num_samples)

    neg_spo = torch.cat(
        (torch.repeat_interleave(spo[:, :2], num_samples, dim=0),
         torch.reshape(neg_o, (-1, 1))),
        dim=1)

    return neg_spo


def load_neg_spo(dataset, size='s'):
    """
    :param dataset: kge.dataset.Dataset
    :return: torch.Tensor of negative triples loaded from directory
    """
    negs = []

    for split in ('valid_neg', 'test_neg'):
        triples = pd.read_csv(
            os.path.join('data/triples/codex-' + size, split + '.txt'),
            sep='\t', header=None).values

        # Convert string IDs to integer IDs
        entity_ids = {
            val: i for i, val in enumerate(dataset.entity_ids())}
        relation_ids = {
            val: i for i, val in enumerate(dataset.relation_ids())}

        triples = [
            [entity_ids[s], relation_ids[p], entity_ids[o]]
            for (s, p, o) in triples]

        negs.append(torch.tensor(triples))

    return negs


def main():
    args = parse_args()

    with torch.no_grad():
        # Load first model, get dataset
        # Assumes all models trained on same data
        checkpoint = load_checkpoint(args.model_files[0])
        model = kge.model.KgeModel.create_from(checkpoint)
        dataset = model.dataset

        splits = ('valid', 'test')
        valid_spo, test_spo = [
            dataset.split(split) for split in splits]

        if args.negative in ('uniform', 'frequency'):
            valid_neg_spo, test_neg_spo = [
                generate_neg_spo(dataset, split,
                                 negative_type=args.negative_type)
                for split in splits]
        else:
            valid_neg_spo, test_neg_spo = load_neg_spo(dataset, size=args.size)
            print('Loaded {} valid negatives and {} test negatives'.format(
                len(valid_neg_spo), len(test_neg_spo)))

        metrics = []
        for model_file in args.model_files:
            if os.path.exists(model_file):
                checkpoint = load_checkpoint(model_file)
                model = kge.model.KgeModel.create_from(checkpoint)

                # Score negative and positive validation triples
                X_valid, y_valid = get_X_y(
                    model, valid_spo, valid_neg_spo)
                X_test, y_test = get_X_y(
                    model, test_spo, test_neg_spo)

                # Calibrate scores and predict on valid and test sets
                calibrator = CalibratedClassifierCV(
                    cv=5, method=args.calib_type)
                calibrator.fit(X_valid, y_valid)

                Xs, ys = (X_valid, X_test), (y_valid, y_test)
                metrics.append({
                    'model_file': model_file
                })
                for X, y, split in zip(Xs, ys, splits):
                    y_pred = calibrator.predict(X)
                    metrics[-1]['accuracy_' + split] = (
                        accuracy_score(y, y_pred))
                    metrics[-1]['f1_' + split] = f1_score(y, y_pred)

        for metric in metrics:
            for key, val in metric.items():
                print('{}: {}'.format(key, val))
            print()


if __name__ == '__main__':
    main()
