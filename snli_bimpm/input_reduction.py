import json
import numpy as np
from copy import deepcopy
from typing import List

import torchtext
from nltk import word_tokenize

import allennlp_models.nli
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.predictors import Predictor


def real_sequence_length(text_field: TextField, ignore_tokens: List[str] = ['@@NULL@@']):
    return len([x for x in text_field.tokens if x.text not in ignore_tokens])


def real_text(text_field: TextField, ignore_tokens: List[str] = ['@@NULL@@']):
    return ' '.join(x.text for x in text_field.tokens if x.text not in ignore_tokens)


def remove_one_token(
        predictor: Predictor,
        instances: List[Instance],
        reduction_field_name: str,
        gradient_name: str,
        n_beams: List[int],
        indices: List[List[int]],
        removed_indices: List[List[int]],
        max_beam_size: int = 5,
        min_sequence_length: int = 1,
        ignore_tokens: List[str] = ['@@NULL@@'],
):
    """
    remove one token from each example.
    each example branches out to at most max_beam_size new beams.
    we do not do beam verification here.

    batch structure:
    > example 0 beam 1
    > example 0 beam 2  # n_beams[0] = 2
    > example 1 beam 1  # n_beams[1] = 1
    > example 2 beam 1
    > example 2 beam 2  # n_beams[2] = 2
    >                   # n_beams[3] = 0

    """
    n_examples = len(n_beams)  # not batch size!

    if 'label' not in instances[0].fields:
        outputs = predictor.predict_batch_instance(instances)
        instances = [predictor.predictions_to_labeled_instances(i, o)[0] for i, o in zip(instances, outputs)]

    # one forward-backward pass to get the score of each token in the batch
    gradients, outputs = predictor.get_gradients(instances)
    grads = gradients[gradient_name]
    onehot_grad = np.einsum('bld,bld->bl', grads, grads)

    # beams of example_idx: batch[start: start + n_beams[example_idx]]
    start = 0
    new_instances = []
    new_n_beams = [0 for _ in range(n_examples)]
    new_indices = []
    new_removed_indices = []
    current_lengths = [real_sequence_length(x[reduction_field_name]) for x in instances]

    for example_idx in range(n_examples):
        """
        for each example_idx, current beams -> future beams
        1. find beam-level reduction candidates
        2. merge and sort them to get example-level reduction candidates
        """
        # skip if example_idx exited the search
        if n_beams[example_idx] == 0:
            continue

        # find beam-level candidates
        candidates = []  # (batch_index i, token j)
        for i in range(start, start + n_beams[example_idx]):
            if current_lengths[i] <= min_sequence_length:
                # nothing to reduce
                continue
            beam_candidates = [
                (i, j) for j in np.argsort(- onehot_grad[i])
                if (
                    j < instances[i][reduction_field_name].sequence_length()
                    and instances[i][reduction_field_name].tokens[j].text not in ignore_tokens
                )
            ]
            candidates += beam_candidates[:max_beam_size]

        # no beam-level candidate found, skip
        if len(candidates) == 0:
            start += n_beams[example_idx]
            continue

        # gather scores of all example-level candidates
        # sort them to get example-level candidates
        candidates = np.asarray(candidates)
        scores = onehot_grad[candidates[:, 0], candidates[:, 1]]
        candidate_scores = sorted(zip(candidates, scores), key=lambda x: -x[1])
        candidates = [candidate for candidate, score in candidate_scores[:max_beam_size]]

        # each candidate should be a valid token in the beam it belongs
        assert all(j < current_lengths[i] for i, j in candidates)

        for i, j in candidates:
            new_instance = deepcopy(instances[i])
            tokens = new_instance[reduction_field_name].tokens
            new_instance[reduction_field_name].tokens = tokens[0: j] + tokens[j + 1:]
            new_instance.indexed = False

            new_n_beams[example_idx] += 1
            new_instances.append(new_instance)
            new_removed_indices.append(removed_indices[i] + [indices[i][j]])
            new_indices.append(indices[i][:j] + indices[i][j + 1:])

        # move starting position to next example
        start += n_beams[example_idx]

    return new_instances, new_n_beams, new_indices, new_removed_indices


def reduce_instances(
        predictor: Predictor,
        instances: List[Instance],
        reduction_field_name: str,
        gradient_name: str,
        max_beam_size: int = 5,
        prob_threshold: float = -1,
        min_sequence_length: int = 1,
        ignore_tokens: List[str] = ['@@NULL@@'],
):
    """
    original batch
    > example 0
    > example 1
    > example 2
    > example 3

    during reduction, and example 4 already exited the search
    > example 0 beam 1
    > example 0 beam 2  # n_beams[0] = 2
    > example 1 beam 1  # n_beams[1] = 1
    > example 2 beam 1
    > example 2 beam 2  # n_beams[2] = 2
    >                   # n_beams[3] = 0


    then each example i beam j branches out to
    > example i beam j 0
    > example i beam j 1
    > ...

    which forms
    > example i beam j 0
    > example i beam j 1
    > example i beam j 2
    > example i beam k 0
    > example i beam k 1

    we sort all beams of example i, select the top ones, filter, and go to next step

    :param predictor:
    :param instances:
    :param reduction_field_name:
    :param gradient_name:
    :param max_beam_size:
    :param prob_threshold:
    """

    if 'label' not in instances[0].fields:
        outputs = predictor.predict_batch_instance(instances)
        instances = [predictor.predictions_to_labeled_instances(i, o)[0] for i, o in zip(instances, outputs)]

    n_examples = len(instances)
    n_beams = [1 for _ in range(n_examples)]  # each example starts with 1 beam
    indices = [
        [
            i for i, token in enumerate(instance[reduction_field_name])
            if token.text not in ignore_tokens
        ]
        for instance in instances
    ]
    removed_indices = [[] for _ in range(n_examples)]

    # keep track of a single shortest reduced versions
    shortest_instances = {i: deepcopy(x) for i, x in enumerate(instances)}
    shortest_lengths = {
        i: real_sequence_length(x[reduction_field_name], ignore_tokens)
        for i, x in enumerate(instances)
    }
    shortest_removed_indices = {}

    # to make sure predictions remain the same
    original_instances = deepcopy(instances)

    while True:
        # all beams are reduced at the same pace
        # next step beam size from each current example is at most its number of tokens

        # remove one token from each example
        instances, n_beams, indices, removed_indices = remove_one_token(
            predictor,
            instances,
            reduction_field_name=reduction_field_name,
            gradient_name=gradient_name,
            n_beams=n_beams,
            indices=indices,
            removed_indices=removed_indices,
            max_beam_size=max_beam_size,
            min_sequence_length=min_sequence_length,
            ignore_tokens=ignore_tokens,
        )

        # verify prediction for each beam
        outputs = predictor.predict_batch_instance(instances)

        # beams of example_idx: batch[start: start + n_beams[example_idx]]
        start = 0
        new_instances = []
        new_indices = []
        new_n_beams = [0 for _ in range(n_examples)]
        new_removed_indices = []
        current_lengths = [real_sequence_length(x[reduction_field_name]) for x in instances]

        for example_idx in range(n_examples):
            original_field = original_instances[example_idx][reduction_field_name]
            original_length = real_sequence_length(original_field, ignore_tokens)
            for i in range(start, start + n_beams[example_idx]):
                assert current_lengths[i] + len(removed_indices[i]) == original_length
                reduced_prediction = np.argmax(outputs[i]['label_probs'])
                reduced_score = outputs[i]['label_probs'][reduced_prediction]
                original_prediction = original_instances[example_idx]['label'].label
                if (
                        reduced_prediction == original_prediction
                        and reduced_score >= prob_threshold
                ):
                    # check if this new valid reduced example is shorter than current
                    # reduced_token_sequence = instances[i][reduction_field_name].tokens
                    if current_lengths[i] < shortest_lengths[example_idx]:
                        shortest_instances[example_idx] = deepcopy(instances[i])
                        # shortest_token_sequences[example_idx] = [reduced_token_sequence]
                        shortest_removed_indices[example_idx] = removed_indices[i]
                        shortest_lengths[example_idx] = current_lengths[i]
                    # elif (
                    #     current_lengths[i] == shortest_lengths[example_idx]
                    #     and reduced_token_sequence not in shortest_token_sequences[example_idx]
                    # ):
                    #     shortest_instances[example_idx].append(deepcopy(instances[i]))
                    #     shortest_token_sequences[example_idx].append(reduced_token_sequence)
                    #     shortest_removed_indices[example_idx].append(removed_indices[i])

                    if current_lengths[i] <= min_sequence_length:
                        # all beams of an example has the same length
                        # this means all beams of this example has length 1
                        # do not branch out from this example
                        pass
                    else:
                        # beam valid, but not short enough, keep reducing
                        new_n_beams[example_idx] += 1
                        new_instances.append(instances[i])
                        new_indices.append(indices[i])
                        new_removed_indices.append(removed_indices[i])

            # move cursor to next example then update the beam count of this example
            start += n_beams[example_idx]

        if len(new_instances) == 0:
            break

        instances = new_instances
        n_beams = new_n_beams
        indices = new_indices
        removed_indices = new_removed_indices

    shortest_instances = [shortest_instances[i] for i in range(n_examples)]
    shortest_removed_indices = [shortest_removed_indices.get(i, []) for i in range(n_examples)]
    return shortest_instances, shortest_removed_indices


def main():
    predictor = Predictor.from_path(
        'https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz',
        predictor_name='textual-entailment',
        cuda_device=0,
    )

    print('loading data from {}'.format('data/'))
    train, dev, test = torchtext.datasets.SNLI.splits(
        torchtext.data.Field(batch_first=True, tokenize=word_tokenize, lower=True),
        torchtext.data.Field(sequential=False, unk_token=None),
        root='data/')
    dataset = dev

    batch_size = 10
    checkpoint = []
    for batch_start in range(0, len(dataset), batch_size):

        if batch_start > 30:
            break

        examples = dataset[batch_start: batch_start + batch_size]
        n_examples = len(examples)
        inputs = [{
            'premise': ' '.join(x.premise),
            'hypothesis': ' '.join(x.hypothesis),
            'label': x.label,
        } for x in examples]

        instances = predictor._batch_json_to_instances(inputs)
        outputs = predictor.predict_batch_instance(instances)
        instances = [predictor.predictions_to_labeled_instances(i, o)[0] for i, o in zip(instances, outputs)]

        reduced_instances, removed_indices = reduce_instances(
            predictor,
            instances,
            reduction_field_name='hypothesis',
            gradient_name='grad_input_1',
            max_beam_size=5,
        )

        reduced_instances = [x[0] for x in reduced_instances]
        removed_indices = [x[0] for x in removed_indices]

        for example_idx in range(n_examples):
            reduced_instance = reduced_instances[example_idx]
            removed = removed_indices[example_idx]
            original_example = inputs[example_idx]

            reduced_example = {
                'premise': real_text(reduced_instance['premise']),
                'hypothesis': real_text(reduced_instance['hypothesis']),
                'label': original_example['label'],
            }

            checkpoint.append({
                'original': original_example,
                'reduced': reduced_example,
                'removed_indices': removed,
            })

#         if batch_i % 1000 == 0 and batch_i > 0:
#             out_path = os.path.join(out_dir, '{}.{}'.format(fname, batch_i))
#             with open(out_path, 'wb') as f:
#                 pickle.dump(checkpoint, f)
#             checkpoint = []
#
#     if len(checkpoint) > 0:
#         out_path = os.path.join(out_dir, '{}.{}'.format(fname, batch_i))
#         with open(out_path, 'wb') as f:
#             pickle.dump(checkpoint, f)
    with open('reduced_dev.json', 'w') as f:
        json.dump(checkpoint, f)


if __name__ == '__main__':
    main()
