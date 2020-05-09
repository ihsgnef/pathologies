import json
import numpy as np
from copy import deepcopy
from typing import List, Dict

import torchtext
from nltk import word_tokenize
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.predictors import Predictor
import allennlp_models.nli
import allennlp_models.sentiment


def real_sequence_length(text_field: TextField, ignore_tokens: List[str] = ['@@NULL@@']):
    return len([x for x in text_field.tokens if x.text not in ignore_tokens])


def real_text(text_field: TextField, ignore_tokens: List[str] = ['@@NULL@@']):
    return ' '.join(x.text for x in text_field.tokens if x.text not in ignore_tokens)


def replace_one_token(
        predictor: Predictor,
        instances: List[Instance],
        reduction_field_name: str,
        gradient_field_name: str,
        n_beams: List[int],
        indices: List[List[int]],
        replaced_indices: List[List[int]],
        embedding_weight: np.ndarray,
        index_to_token: Dict[int, str],
        max_beam_size: int = 5,
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
        instances = [predictor.predictions_to_labeled_instances(i, o)[0]
                     for i, o in zip(instances, outputs)]

    # one forward-backward pass to get the score of each token in the batch
    gradients, outputs = predictor.get_gradients(instances)
    grads = gradients[gradient_field_name]
    hotflip_grad = np.einsum('bld,kd->blk', grads, embedding_weight)
    sign = -1

    # beams of example_idx: batch[start: start + n_beams[example_idx]]
    start = 0
    new_instances = []
    new_n_beams = [0 for _ in range(n_examples)]
    new_indices = []
    new_replaced_indices = []
    current_lengths = [real_sequence_length(x[reduction_field_name], ignore_tokens)
                       for x in instances]

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
        candidates = []  # (batch_index i, token j, replacement k)
        for i in range(start, start + n_beams[example_idx]):
            field = instances[i][reduction_field_name]
            # argsort the flattened scores
            indices_sorted = np.argsort(sign * hotflip_grad[i].ravel())
            # unravel into original shape
            indices_sorted = np.unravel_index(indices_sorted, hotflip_grad[i].shape)
            indices_sorted = np.stack(indices_sorted, 1)
            beam_candidates = [
                (i, j, k) for j, k in indices_sorted 
                if (
                    j < field.sequence_length()
                    and field.tokens[j].text not in ignore_tokens
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
        scores = hotflip_grad[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
        candidate_scores = sorted(zip(candidates, scores), key=lambda x: sign * x[1])
        candidates = [c for c, s in candidate_scores[:max_beam_size]]

        # each candidate should be a valid token in the beam it belongs
        assert all(j < current_lengths[i] for i, j, k in candidates)

        for i, j, k in candidates:
            new_instance = deepcopy(instances[i])
            new_instance[reduction_field_name].tokens = (
                new_instance[reduction_field_name].tokens[0: j]
                + [Token(index_to_token[k])]
                + new_instance[reduction_field_name].tokens[j + 1:]
            )
            new_instance.indexed = False

            new_n_beams[example_idx] += 1
            new_instances.append(new_instance)
            new_replaced_indices.append(replaced_indices[i] + [indices[i][j]])
            new_indices.append(indices[i])

        # move starting position to next example
        start += n_beams[example_idx]

    return new_instances, new_n_beams, new_indices, new_replaced_indices


def replace_instances(
        predictor: Predictor,
        instances: List[Instance],
        reduction_field_name: str,
        gradient_field_name: str,
        probs_field_name: str,
        embedding_weight: np.ndarray,
        index_to_token: Dict[int, str],
        max_beam_size: int = 5,
        max_replace_steps: int = 5,
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

    we sort all beams of example i, select the top ones,
    filter ones that do not retain prediction, go to next step

    :param predictor:
    :param instances:
    :param reduction_field_name:
    :param gradient_field_name:
    :param probs_field_name:
    :param max_beam_size:
    """

    if 'label' not in instances[0].fields:
        outputs = predictor.predict_batch_instance(instances)
        instances = [predictor.predictions_to_labeled_instances(i, o)[0]
                     for i, o in zip(instances, outputs)]

    n_examples = len(instances)
    n_beams = [1 for _ in range(n_examples)]  # each example starts with 1 beam
    indices = [[
        i for i, token in enumerate(instance[reduction_field_name])
        if token.text not in ignore_tokens
    ] for instance in instances]
    replaced_indices = [[] for _ in range(n_examples)]

    # perturbed instances that do not necessarily flip model prediction
    final_instances = {i: deepcopy(instance) for i, instance in enumerate(instances)}
    final_replaced_indices = {i: [] for i in range(n_examples)}

    # check if prediction is flipped
    original_instances = deepcopy(instances)

    for _ in range(max_replace_steps):
        # all beams are reduced at the same pace
        # remove one token from each example
        instances, n_beams, indices, replaced_indices = replace_one_token(
            predictor,
            instances,
            reduction_field_name=reduction_field_name,
            gradient_field_name=gradient_field_name,
            n_beams=n_beams,
            indices=indices,
            replaced_indices=replaced_indices,
            embedding_weight=embedding_weight,
            index_to_token=index_to_token,
            max_beam_size=max_beam_size,
            ignore_tokens=ignore_tokens,
        )

        # verify prediction for each beam
        outputs = predictor.predict_batch_instance(instances)

        # beams of example_idx: batch[start: start + n_beams[example_idx]]
        start = 0
        new_instances = []
        new_indices = []
        new_replaced_indices = []
        new_n_beams = [0 for _ in range(n_examples)]
        for example_idx in range(n_examples):
            for i in range(start, start + n_beams[example_idx]):
                reduced_prediction = np.argmax(outputs[i][probs_field_name])
                original_prediction = original_instances[example_idx]['label'].label
                final_instances[example_idx] = deepcopy(instances[i])
                final_replaced_indices[example_idx] = replaced_indices[i]
                if reduced_prediction == original_prediction:
                    # prediction not flipped yet, keep replacing
                    new_n_beams[example_idx] += 1
                    new_instances.append(instances[i])
                    new_indices.append(indices[i])
                    new_replaced_indices.append(replaced_indices[i])

            # move cursor to next example then update the beam count of this example
            start += n_beams[example_idx]

        if len(new_instances) == 0:
            break

        instances = new_instances
        n_beams = new_n_beams
        indices = new_indices
        replaced_indices = new_replaced_indices

    return (
        [final_instances[i] for i in range(n_examples)],
        [final_replaced_indices[i] for i in range(n_examples)]
    )


def snli():
    predictor = Predictor.from_path(
        'https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz',
        predictor_name='textual-entailment',
        cuda_device=0,
    )

    train, dev, test = torchtext.datasets.SNLI.splits(
        torchtext.data.Field(batch_first=True, tokenize=word_tokenize, lower=False),
        torchtext.data.Field(sequential=False, unk_token=None),
        root='data/')
    dataset = dev

    ignore_tokens = ["@@NULL@@"]
    reduction_field_name = 'hypothesis'
    gradient_field_name = 'grad_input_1'
    probs_field_name = 'label_probs'

    batch_size = 10
    checkpoint = []
    for batch_start in range(0, len(dataset), batch_size):

        if batch_start > 30:
            break

        inputs = [{
            'premise': ' '.join(x.premise),
            'hypothesis': ' '.join(x.hypothesis),
            'label': x.label,
        } for x in dataset[batch_start: batch_start + batch_size]]
        original_instances = predictor._batch_json_to_instances(inputs)
        original_outputs = predictor.predict_batch_instance(original_instances)
        instances = [predictor.predictions_to_labeled_instances(i, o)[0]
                     for i, o in zip(original_instances, original_outputs)]

        reduced_instances, removed_indices = reduce_instances(
            predictor,
            instances,
            reduction_field_name=reduction_field_name,
            gradient_field_name=gradient_field_name,
            probs_field_name=probs_field_name,
            max_beam_size=5,
            ignore_tokens=ignore_tokens,
        )

        reduced_outputs = predictor.predict_batch_instance(reduced_instances)
        original_predictions = [np.argmax(x[probs_field_name]) for x in original_outputs]
        reduced_predictions = [np.argmax(x[probs_field_name]) for x in reduced_outputs]
        assert original_predictions == reduced_predictions

        for example_idx, original_instance in enumerate(inputs):
            checkpoint.append({
                'original': original_instance,
                'reduced': {
                    'premise': real_text(reduced_instances[example_idx]['premise'], ignore_tokens),
                    'hypothesis': real_text(reduced_instances[example_idx]['hypothesis'], ignore_tokens),
                    'label': original_instance['label'],
                },
                'removed_indices': removed_indices[example_idx],
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


def sst():
    predictor = Predictor.from_path(
        'https://s3-us-west-2.amazonaws.com/allennlp/models/sst-2-basic-classifier-glove-2019.06.27.tar.gz',
        predictor_name='text_classifier',
        cuda_device=0,
    )

    train, dev, test = torchtext.datasets.SST.splits(
        torchtext.data.Field(batch_first=True, tokenize=word_tokenize, lower=False),
        torchtext.data.Field(sequential=False, unk_token=None),
        root='data/')
    dataset = dev

    batch_size = 10
    checkpoint = []
    for batch_start in range(0, len(dataset), batch_size):

        if batch_start > 30:
            break

        inputs = [{
            'sentence': ' '.join(x.text),
            'label': x.label,
        } for x in dataset[batch_start: batch_start + batch_size]]
        n_examples = len(inputs)
        original_instances = predictor._batch_json_to_instances(inputs)
        original_outputs = predictor.predict_batch_instance(original_instances)
        instances = [predictor.predictions_to_labeled_instances(i, o)[0]
                     for i, o in zip(original_instances, original_outputs)]

        reduced_instances, removed_indices = reduce_instances(
            predictor,
            instances,
            reduction_field_name='tokens',
            gradient_field_name='grad_input_1',
            probs_field_name='probs',
            max_beam_size=5,
        )

        for example_idx in range(n_examples):
            checkpoint.append({
                'original': inputs[example_idx],
                'reduced': {
                    'sentence': real_text(reduced_instances[example_idx]['tokens']),
                    'label': inputs[example_idx]['label'],
                },
                'removed_indices': removed_indices[example_idx],
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


def squad():
    single_id = SingleIdTokenIndexer(lowercase_tokens=True)
    reader = SquadReader(token_indexers={'tokens': single_id})
    dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json')
    predictor = Predictor.from_path(
        'https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-glove-2019.05.09.tar.gz',
        predictor_name='reading-comprehension',
    )


if __name__ == '__main__':
    sst()
