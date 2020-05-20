"""
Triple classification using different
negative generation strategies and calibration techniques
"""
import argparse
import os
import torch

import pandas as pd

from sklearn.metrics import (
    brier_score_loss, classification_report)
from sklearn.calibration import CalibratedClassifierCV

import kge.model
import kge.config
import kge.util.sampler


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_files', nargs='+',
        help='LibKGE model checkpoint(s)'
    )

    parser.add_argument(
        '--valid-neg', default='data/triples/negatives/valid.txt',
        help=(
            'File of validation negatives. '
            'For true_neg negative type only.'
        )
    )

    parser.add_argument(
        '--test-neg', default='data/triples/negatives/test.txt',
        help=(
            'File of test negatives. '
            'For true_neg negative type only.'
        )
    )

    parser.add_argument(
        '--negative-types', default=['uniform'],
        nargs='+',
        choices=['frequency', 'uniform', 'true_neg'],
        help=(
            'Types of negative sampling to use. '
            'Default is [uniform].'
        )
    )

    parser.add_argument(
        '--calib-type', default='sigmoid',
        choices=['sigmoid', 'isotonic'],
        help=(
            'Calibrator type. '
            'Default is sigmoid (Platt scaling)'
        )
    )

    parser.add_argument(
        '--num-samples', default=1, type=int,
        help=(
            'Number of negatives per positive. '
            'Default 1'
        )
    )

    parser.add_argument(
        '--csv', default=None,
        help=(
            'CSV filename to save results. '
            'Default None; if an argument is provided, '
            'writes results to the specified file.'
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


def load_neg_spo(dataset, fname):
    """
    :param dataset: kge.dataset.Dataset
    :param fname: filename of negatives to load
    :return: torch.Tensor of negative triples loaded from directory
    """
    negs = pd.read_csv(
        os.path.join(fname), sep='\t', header=None).values

    # Convert string IDs to integer IDs
    entity_ids = {
        val: i for i, val in enumerate(dataset.entity_ids())}
    relation_ids = {
        val: i for i, val in enumerate(dataset.relation_ids())}

    triples = []
    for s, p, o in negs:
        if (s in entity_ids and
                p in relation_ids and
                o in entity_ids):
            triples.append([
                entity_ids[s], relation_ids[p], entity_ids[o]])

    return torch.tensor(triples)


def triple_classification(
        model_files,
        valid_neg=None,
        test_neg=None,
        negative_type='uniform',
        num_samples=1,
        calib_type='sigmoid'):
    """
    :param model_files: model files to test on
    :param valid_neg: filename of valid negatives to load
    :param test_neg: filename of test negatives to load
    :param negative_type: one of 'uniform', 'frequency'
    :param num_samples: number of negatives per positive to generate
        Only for random negative generation
    :param calib_type: One of 'sigmoid', 'isotonic'
    :return metrics: dictionary of performance metrics/metadata
    """
    print('Testing on', len(model_files), 'models')

    # Load first model, get dataset
    # Assumes all models trained on same data
    model = kge.model.KgeModel.load_from_checkpoint(model_files[0])
    dataset = model.dataset

    splits = ('valid', 'test')
    valid_spo, test_spo = [
        dataset.split(split) for split in splits]

    if negative_type in ('uniform', 'frequency'):
        valid_neg_spo, test_neg_spo = [
            generate_neg_spo(dataset, split,
                             negative_type=negative_type,
                             num_samples=num_samples)
            for split in splits]
    elif (negative_type == 'true_neg' and
          valid_neg is not None and
          test_neg is not None
          ):
        valid_neg_spo, test_neg_spo = [
            load_neg_spo(dataset, split)
            for split in (valid_neg, test_neg)]
        print('Loaded {} valid negatives and {} test negatives'.format(
            len(valid_neg_spo), len(test_neg_spo)))
    else:
        raise ValueError('Invalid negative sampling type')

    metrics = []
    for model_file in model_files:
        if os.path.exists(model_file):
            model = kge.model.KgeModel.load_from_checkpoint(model_file)
            print('Testing on', model_file)

            # Score negative and positive validation triples
            X_valid, y_valid = get_X_y(
                model, valid_spo, valid_neg_spo)
            X_test, y_test = get_X_y(
                model, test_spo, test_neg_spo)

            # Calibrate scores and predict on valid and test sets
            calibrator = CalibratedClassifierCV(cv=5, method=calib_type)
            calibrator.fit(X_valid, y_valid)

            Xs, ys = (X_valid, X_test), (y_valid, y_test)
            for X, y, split in zip(Xs, ys, splits):
                pos_proba = calibrator.predict_proba(X)[:, 1]
                y_pred = calibrator.predict(X)
                report = classification_report(y, y_pred, output_dict=True)
                report = report['0.0']

                metrics.append({
                    'model_file': model_file,
                    'split': split,
                    'negative_type': negative_type,
                    'valid_neg': valid_neg,
                    'test_neg': test_neg,
                    'calib_type': calib_type,
                    'num_samples': num_samples,
                    'brier_score': brier_score_loss(y, pos_proba),
                    'precision': report['precision'],
                    'recall': report['recall'],
                    'f1': report['f1-score']
                })

                print(split, 'split')
                for key, val in metrics[-1].items():
                    if isinstance(val, float):
                        print('{}: {:.4f}'.format(key, val))

    return metrics


def main():
    args = parse_args()
    dfs = []

    with torch.no_grad():
        for negative_type in args.negative_types:
            metrics = triple_classification(
                args.model_files,
                num_samples=args.num_samples,
                valid_neg=args.valid_neg,
                test_neg=args.test_neg,
                negative_type=negative_type,
                calib_type=args.calib_type)

            if args.csv is not None:
                for data in metrics:
                    dfs.append(pd.DataFrame.from_dict(
                        data, orient='index').transpose())

        if args.csv is not None:
            pd.concat(dfs).to_csv(args.csv, index=False)
            print('Saved triple classification results to', args.csv)


if __name__ == '__main__':
    main()