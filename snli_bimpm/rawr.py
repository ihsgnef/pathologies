import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bimpm import BIMPM
from dataset import SNLI
from util import prepare_output_dir
from args import conf, rawr_conf


class Batch:

    def __init__(self, premise=None, hypothesis=None, label=None):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label = label


def real_length(x):
    # length of vector without padding
    return sum(x != 1)


def get_onehot_grad(model, batch, p_not_h=False):
    criterion = nn.CrossEntropyLoss()
    extracted_grads = {}

    def hook(grad):
        extracted_grads['embed'] = grad

    batch_size, length = batch.premise.shape if p_not_h else batch.hypothesis.shape
    model.train()  # turn on train mode here but we skip dropout with no_dropout in the foward call
    output = model(batch.premise, batch.hypothesis, embed_grad_hook=hook, p_not_h=p_not_h, no_dropout=True)
    label = torch.max(output, 1)[1]
    loss = criterion(output, label)
    loss.backward()

    embed = model.word_emb(batch.premise if p_not_h else batch.hypothesis)
    onehot_grad = embed.view(-1) * extracted_grads['embed'].contiguous().view(-1)
    onehot_grad = onehot_grad.view(batch_size, length, -1).sum(-1)
    return onehot_grad


def to_text(x, vocab):
    if isinstance(x, torch.cuda.LongTensor):
        x = x.cpu()
    if isinstance(x, torch.LongTensor):
        x = x.numpy().tolist()
    return ' '.join(vocab[w] for w in x if w != 1)


def remove_one(
        model,
        batch: Batch,
        n_beams: List[int],
        indices: List[List[int]],
        removed_indices: List[List[int]],
        max_beam_size: int,
        p_not_h=False
) -> Tuple(
    Batch,
    List[int],
    List[List[int]],
    List[List[int]],
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

    :param model: model to perform reduction
    :param batch: batch of examples to reduce
    :param n_beams: number of beams of each example
    :param indices: remaining token indices of each beam
    :param removed_indices: removed token indices of each beam
    :param max_beam_size: max number of beams from each current beam
    :param p_not_h: reduce premise instead of hypothesis
    :return batch: the new batch with reduction done
    :return new_n_beams: number of beams of each example
    :return new_indices: remaining token indices of each beam
    :return new_removed_indices: removed token indices of each beam
    """
    n_examples = len(n_beams)  # not batch size!
    # one forward-backward pass to get the score of each token in the batch
    onehot_grad = get_onehot_grad(model, batch, p_not_h).detach().cpu().numpy()

    # s2 is the one being reduced
    if p_not_h:
        # reduce premise
        s1 = batch.hypothesis
        s2 = batch.premise
    else:
        # reduce hypothesis
        s1 = batch.premise
        s2 = batch.hypothesis

    start = 0  # beams of example_idx: batch[start: start + n_beams[example_idx]]
    new_s1 = []
    new_s2 = []
    new_n_beams = [0 for _ in range(n_examples)]
    # to keep track of tokens removed
    new_indices = []
    new_removed_indices = []
    real_lengths = [real_length(x) for x in s2]

    for example_idx in range(n_examples):
        """
        example_idx: current beams -> future beams
        1. find beam-level reduction candidates
        2. merge and sort them to get example-level reduction candidates
        """
        if n_beams[example_idx] == 0:
            # skil if example_idx exited the search
            start += 0
            continue

        # find beam-level candidates (batch_index i, token j)
        coordinates = []
        for i in range(start, start + n_beams[example_idx]):
            if real_lengths[i] <= 1:
                continue
            order = np.argsort(- onehot_grad[i][:real_lengths[i]])
            coordinates += [(i, j) for j in order[:max_beam_size]]

        # no beam-level candidate found, skip
        if len(coordinates) == 0:
            start += n_beams[example_idx]
            continue

        # gather scores of beam-level candidates
        # meger and sort them to get example-level candidates
        coordinates = np.asarray(coordinates)
        scores = onehot_grad[coordinates[:, 0], coordinates[:, 1]]
        scores = sorted(zip(coordinates, scores), key=lambda x: -x[1])
        coordinates = [x for x, _ in scores[:max_beam_size]]

        # each candidate should be a valid token in the beam it belongs
        assert all(j < real_lengths[i] for i, j in coordinates)

        for i, j in coordinates:
            partial_s2 = []
            if j > 0:
                partial_s2.append(s2[i][:j])
            if j + 1 < s2[i].shape[0]:
                partial_s2.append(s2[i][j + 1:])
            if len(partial_s2) > 0:
                new_s2.append(torch.cat(partial_s2, 0))
                new_s1.append(s1[i])
                new_removed_indices.append(removed_indices[i] + [indices[i][j]])
                new_indices.append(indices[i][:j] + indices[i][j + 1:])
                new_n_beams[example_idx] += 1

        # move starting position to next example
        start += n_beams[example_idx]

    if p_not_h:
        # premise == s2 is being removed
        batch = Batch(torch.stack(new_s2, 0), torch.stack(new_s1, 0))
    else:
        batch = Batch(torch.stack(new_s1, 0), torch.stack(new_s2, 0))
    return batch, new_n_beams, new_indices, new_removed_indices


def get_rawr(
        model,
        batch: Batch,
        original_predictions: np.ndarray = None,
        max_beam_size: int = 5,
        conf_threshold: float = -1,
        p_not_h: bool = False
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

    :param p_not_h: reduce premise instead of hypothesis
    :param conf_threshold: (lower) threshold on confidence
    """

    if original_predictions is None:
        output = model(batch.premise, batch.hypothesis)
        original_predictions = torch.max(output, 1)[1].detach().cpu().numpy()

    batch = Batch(batch.premise, batch.hypothesis, batch.label)
    n_examples = batch.hypothesis.shape[0]
    n_beams = [1 for _ in range(n_examples)]

    # s2 is being reduced
    if p_not_h:
        s1, s2 = batch.hypothesis, batch.premise
    else:
        s1, s2 = batch.premise, batch.hypothesis

    # each token has an (i, j) index
    indices = [list(range(real_length(x))) for x in s2]
    # removed token indices
    removed_indices = [[] for _ in range(n_examples)]

    # keep track of (multiple) shortest reduced versions
    final_s2 = [z for z in s2.detach().cpu().numpy()]
    # remove padding
    final_s2 = [[[int(x) for x in z if x != 1]] for z in final_s2]
    final_removed = [[[]] for _ in range(n_examples)]
    final_length = [len(x) for x in indices]

    while True:
        # all beams are reduced at the same pace
        # next step beam size from each current example is at most its number of tokens
        max_beam_size = min(s2.shape[1], max_beam_size)

        # remove one token from each example
        batch, n_beams, indices, removed_indices = remove_one(
            model, batch, n_beams, indices,
            removed_indices, max_beam_size,
            p_not_h=p_not_h)

        # verify prediction for each beam
        model.eval()
        output = F.softmax(model(batch.premise, batch.hypothesis), 1)
        reduced_scores, reduced_predictions = torch.max(output, 1)
        reduced_scores = reduced_scores.detach().cpu().numpy()
        reduced_predictions = reduced_predictions.detach().cpu().numpy()

        if p_not_h:
            s1, s2 = batch.hypothesis, batch.premise
        else:
            s1, s2 = batch.premise, batch.hypothesis

        start = 0
        new_s1 = []
        new_s2 = []
        new_indices = []
        new_removed = []
        for example_idx in range(n_examples):
            beam_size = 0
            for i in range(start, start + n_beams[example_idx]):
                if (
                        reduced_predictions[i] == original_predictions[example_idx]
                        and reduced_scores[i] >= conf_threshold
                ):
                    new_length = real_length(s2[i])
                    ns2 = s2[i].detach().cpu().numpy()
                    ns2 = [int(x) for x in ns2 if x != 1]

                    # check if this new valid reduced example is shorter than current
                    if new_length == final_length[example_idx]:
                        if ns2 not in final_s2[example_idx]:
                            final_s2[example_idx].append(ns2)
                            final_removed[example_idx].append(removed_indices[i])
                    elif new_length < final_length[example_idx]:
                        final_s2[example_idx] = [ns2]
                        final_removed[example_idx] = [removed_indices[i]]
                        final_length[example_idx] = new_length
                    if new_length == 1:
                        # all beams of an example has the same length
                        # this means all beams of this example has length 1
                        # do not branch out from this example
                        beam_size = 0
                    else:
                        # beam valid, but not short enough, keep reducing
                        beam_size += 1
                        new_s2.append(s2[i])
                        new_s1.append(s1[i])
                        new_indices.append(indices[i])
                        new_removed.append(removed_indices[i])
            # move cursor to next example then update the beam count of this example
            start += n_beams[example_idx]
            n_beams[example_idx] = beam_size

        if len(new_s2) == 0:
            break

        new_s1 = torch.stack(new_s1, 0)
        new_s2 = torch.stack(new_s2, 0)
        if p_not_h:
            batch = Batch(new_s2, new_s1)
        else:
            batch = Batch(new_s1, new_s2)
        indices = new_indices
        removed_indices = new_removed
    return final_s2, final_removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--baseline', default='results/baseline.pt')
    parser.add_argument('--pnoth', default=False, action='store_true',
                        help='reduce premise instead of hypothesis')
    parser.add_argument('--truth', default=False, action='store_true',
                        help='use label instead of prediction as target')
    args = parser.parse_args()

    data = SNLI(conf)
    conf.char_vocab_size = len(data.char_vocab)
    conf.word_vocab_size = len(data.TEXT.vocab)
    conf.class_size = len(data.LABEL.vocab)
    conf.max_word_len = data.max_word_len
    q_vocab = data.TEXT.vocab.itos
    a_vocab = data.LABEL.vocab.itos

    out_dir = prepare_output_dir(conf, 'results', 'rawr')
    print('Generating [{}] rawr data from [{}].'.format(
        args.fold, args.baseline))
    print(out_dir)

    model = BIMPM(conf, data)
    model.load_state_dict(torch.load(args.baseline))
    model.word_emb.weight.requires_grad = True
    model.to(conf.device)

    datasets = {'train': data.train_iter, 'dev': data.dev_iter}

    if args.pnoth:
        fname = 'rawr.{}.premise.pkl'.format(args.fold)
    else:
        fname = 'rawr.{}.hypothesis.pkl'.format(args.fold)

    checkpoint = []
    for batch_i, batch in enumerate(tqdm(datasets[args.fold])):
        if batch_i > len(datasets[args.fold]):
            # otherwise train iter will loop forever!
            break
        batch_size = batch.hypothesis.shape[0]
        model.eval()
        output = F.softmax(model(batch.premise, batch.hypothesis), 1)
        original_scores, original_predictions = torch.max(output, 1)
        original_scores = original_scores.detach().cpu().numpy()
        original_predictions = original_predictions.detach().cpu().numpy()
        batch_cpu = Batch(batch.premise.data.cpu(),
                          batch.hypothesis.data.cpu(),
                          batch.label.data.cpu())

        reduced, removed_indices = get_rawr(
            model, batch, max_beam_size=rawr_conf.max_beam_size,
            conf_threshold=rawr_conf.conf_threshold, p_not_h=args.pnoth)
        for i in range(batch_size):
            og = {
                'premise': batch_cpu.premise[i],
                'hypothesis': batch_cpu.hypothesis[i],
                'premise_readable': to_text(batch_cpu.premise[i], q_vocab),
                'hypothesis_readable': to_text(batch_cpu.hypothesis[i], q_vocab),
                'prediction': original_predictions[i],
                'prediction_readable': a_vocab[original_predictions[i]],
                'score': original_scores[i],
                'label': batch_cpu.label[i],
                'label_readable': a_vocab[batch_cpu.label[i]]
            }
            checkpoint.append({'original': og, 'reduced': []})
            s1 = batch.hypothesis[i] if args.pnoth else batch.premise[i]
            s1 = s1.to(conf.device)
            for j, s2 in enumerate(reduced[i]):
                s2 = torch.LongTensor(s2).to(conf.device)
                model.eval()
                if args.pnoth:
                    output = model(s2.unsqueeze(0), s1.unsqueeze(0))
                else:
                    output = model(s1.unsqueeze(0), s2.unsqueeze(0))
                output = F.softmax(output, 1)
                pred_scores, pred = torch.max(output, 1)
                pred = pred.detach().cpu().numpy()[0]
                pred_scores = pred_scores.detach().cpu().numpy()[0]
                if args.pnoth:
                    hypo, prem = s1.cpu(), s2.cpu()
                else:
                    prem, hypo = s1.cpu(), s2.cpu()
                checkpoint[-1]['reduced'].append({
                    'premise': prem,
                    'hypothesis': hypo,
                    'premise_readable': to_text(prem, q_vocab),
                    'hypothesis_readable': to_text(hypo, q_vocab),
                    'prediction': pred,
                    'prediction_readable': a_vocab[pred],
                    'score': pred_scores,
                    'label': batch_cpu.label[i],
                    'label_readable': a_vocab[batch_cpu.label[i]],
                    'removed_indices': removed_indices[i][j],
                    'which_reduced': 'premise' if args.pnoth else 'hypothesis'
                })
        if batch_i % 1000 == 0 and batch_i > 0:
            out_path = os.path.join(out_dir, '{}.{}'.format(fname, batch_i))
            with open(out_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            checkpoint = []

    if len(checkpoint) > 0:
        out_path = os.path.join(out_dir, '{}.{}'.format(fname, batch_i))
        with open(out_path, 'wb') as f:
            pickle.dump(checkpoint, f)


def padding_tensor(sequences, padding_token=1):
    """
    :param sequences: list of 1D tensors
    :return: padded tensor
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].new(*out_dims).fill_(padding_token)
    mask = sequences[0].new(*out_dims).fill_(padding_token)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor  # , mask


def test():
    data = SNLI(conf)
    conf.char_vocab_size = len(data.char_vocab)
    conf.word_vocab_size = len(data.TEXT.vocab)
    conf.class_size = len(data.LABEL.vocab)
    conf.max_word_len = data.max_word_len

    model = BIMPM(conf, data)
    model.load_state_dict(torch.load('results/baseline.pt'))
    model.word_emb.weight.requires_grad = True
    model = model.to(conf.device).eval()

    batch = next(iter(data.dev_iter))

    output = F.softmax(model(batch.premise, batch.hypothesis), 1)
    original_scores, original_predictions = torch.max(output, 1)
    original_scores = original_scores.detach().cpu().numpy()
    original_predictions = original_predictions.detach().cpu().numpy()

    reduced, removed_indices = get_rawr(
        model, batch,
        max_beam_size=rawr_conf.max_beam_size,
        conf_threshold=rawr_conf.conf_threshold,
        p_not_h=False,
    )

    reduced_hypothesis = padding_tensor([torch.LongTensor(r[0]) for r in reduced])
    reduced_hypothesis = reduced_hypothesis.to(conf.device)
    output = F.softmax(model(batch.premise, batch.hypothesis), 1)
    reduced_scores, reduced_predictions = torch.max(output, 1)
    reduced_scores = reduced_scores.detach().cpu().numpy()
    reduced_predictions = reduced_predictions.detach().cpu().numpy()

    print(all(reduced_predictions == original_predictions))


if __name__ == '__main__':
    main()
