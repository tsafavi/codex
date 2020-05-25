import argparse
import os
import json
import torch

import numpy as np
import pandas as pd

import kge.model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_file',
        help='Directory where model checkpoint is stored'
    )

    parser.add_argument(
        'out_dir',
        help='Directory to save generated negatives'
    )

    parser.add_argument(
        '--type-file', default='data/types/entity2types.json',
        help='Path to entity-type mapping'
    )

    parser.add_argument(
        '--k', default=10, type=int,
        help='Top-k predictions to take as candidate negatives'
    )

    parser.add_argument(
        '--splits', nargs='+', default=['valid', 'test'],
        choices=['train', 'test', 'valid'],
        help='Dataset splits to generate negatives on'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load entity type mapping
    entity_types = json.load(open(args.type_file))

    # Load model checkpoint
    model = kge.model.KgeModel.load_from_checkpoint(args.model_file)

    with torch.no_grad():
        spo = torch.cat(
            [model.dataset.split(name) for name in ('train', 'valid', 'test')],
            dim=0)

        for split_name in args.splits:
            print('Generating negatives on', split_name, 'split')
            split_spo = model.dataset.split(split_name)

            # Get IDs of top-scoring object entities
            scores = model.score_sp(split_spo[:, 0], split_spo[:, 1])
            scores, o_candidates = torch.sort(scores, dim=1)
            scores = torch.flatten(scores[:, -args.k:])
            o_candidates = torch.flatten(o_candidates[:, -args.k:])

            # Concatenate subject-predicates with predicted object entities
            spo_candidates = torch.cat(
                (torch.repeat_interleave(split_spo[:, :2], args.k, dim=0),
                 torch.reshape(o_candidates, (-1, 1))),
                dim=1)

            # Get indices of predictions that are not known positives
            neg_idx = torch.ones_like(o_candidates, dtype=torch.bool)
            for row in spo:
                neg_idx = neg_idx & ~torch.all(spo_candidates == row, dim=1)

            # Map entity integer IDs to string IDs
            candidate_entity_ids = model.dataset.entity_ids(
                indexes=torch.LongTensor(o_candidates))
            entity_ids = np.repeat(
                model.dataset.entity_ids(
                    indexes=torch.LongTensor(split_spo[:, -1])),
                args.k)

            # Get indices of predictions that type-match ground-truth entities
            type_match_idx = torch.BoolTensor([
                len(set(entity_types[eid]).intersection(
                    set(entity_types[candidate_eid]))) > 0
                for eid, candidate_eid in zip(
                    candidate_entity_ids, entity_ids)])

            # Filter out candidates
            spo_candidates = spo_candidates[neg_idx & type_match_idx]
            print('Generated', len(spo_candidates), 'negatives')

            # Convert to Wikidata IDs
            negatives = [
                [model.dataset.entity_ids(indexes=h),
                 model.dataset.relation_ids(indexes=r),
                 model.dataset.entity_ids(indexes=t)]
                for h, r, t in spo_candidates]

            # Save candidates
            out_file = os.path.join(args.out_dir, split_name + '.txt')
            pd.DataFrame.from_records(negatives).to_csv(
                out_file, index=False, header=None, sep='\t')
            print('Saved negatives to', out_file)


if __name__ == '__main__':
    main()
