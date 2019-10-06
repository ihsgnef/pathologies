import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from bimpm import BIMPM
from dataset import SNLI
from util import prepare_output_dir


class Batch:

    def __init__(self, premise=None, hypothesis=None, label=None):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label = label


def real_length(x):
    # length of vector without padding
    if isinstance(x, Variable):
        return sum(x.data != 1)
    else:
        return sum(x != 1)


def get_onehot_grad(model, batch, p_not_h=False):
    criterion = nn.CrossEntropyLoss()
    extracted_grads = {}

    def extract_grad_hook(name):
        def hook(grad):
            extracted_grads[name] = grad
        return hook

    if p_not_h:
        batch_size, length = batch.premise.shape
    else:
        batch_size, length = batch.hypothesis.shape
    batch.premise.volatile = False
    batch.hypothesis.volatile = False
    model.eval()
    output = model(batch.premise, batch.hypothesis,
                   embed_grad_hook=extract_grad_hook('embed'),
                   p_not_h=p_not_h)
    label = torch.max(output, 1)[1]
    loss = criterion(output, label)
    loss.backward()
    embed_grad = extracted_grads['embed']
    if p_not_h:
        embed = model.word_emb(batch.premise)
    else:
        embed = model.word_emb(batch.hypothesis)
    onehot_grad = embed.view(-1) * embed_grad.contiguous().view(-1)
    onehot_grad = onehot_grad.view(batch_size, length, -1).sum(-1)
    return onehot_grad


def remove_one(model, batch, n_beams, indices, removed_indices, max_beam_size,
               p_not_h=False):
    n_examples = len(n_beams)  # not batch size!
    onehot_grad = get_onehot_grad(model, batch, p_not_h).data.cpu().numpy()

    # s2 is the one being reduced
    if p_not_h:
        # reduce premise
        s1 = batch.hypothesis
        s2 = batch.premise
    else:
        # reduce hypothesis
        s1 = batch.premise
        s2 = batch.hypothesis

    start = 0
    new_s1 = []
    new_s2 = []
    new_n_beams = []
    new_indices = []
    new_removed_indices = []
    real_lengths = [real_length(x) for x in s2]

    for example_idx in range(n_examples):
        if n_beams[example_idx] == 0:
            new_n_beams.append(0)
            continue

        coordinates = []
        for i in range(start, start + n_beams[example_idx]):
            if real_lengths[i] <= 1:
                continue
            order = np.argsort(- onehot_grad[i][:real_lengths[i]])
            coordinates += [(i, j) for j in order[:max_beam_size]]

        if len(coordinates) == 0:
            new_n_beams.append(0)
            start += n_beams[example_idx]
            continue

        coordinates = np.asarray(coordinates)
        scores = onehot_grad[coordinates[:, 0], coordinates[:, 1]]
        scores = sorted(zip(coordinates, scores), key=lambda x: -x[1])
        coordinates = [x for x, _ in scores[:max_beam_size]]

        assert all(j < real_lengths[i] for i, j in coordinates)

        cnt = 0
        for i, j in coordinates:
            partial_s2 = []
            if j > 0:
                partial_s2.append(s2[i][:j])
            if j + 1 < s2[i].shape[0]:
                partial_s2.append(s2[i][j + 1:])
            if len(partial_s2) > 0:
                new_s2.append(torch.cat(partial_s2, 0))
                new_s1.append(s1[i])
                new_removed_indices.append(
                        removed_indices[i] + [indices[i][j]])
                new_indices.append(indices[i][:j] + indices[i][j+1:])
                cnt += 1
        new_n_beams.append(cnt)
        start += n_beams[example_idx]

    if p_not_h:
        # premise == s2 is being removed
        batch = Batch(torch.stack(new_s2, 0), torch.stack(new_s1, 0))
    else:
        batch = Batch(torch.stack(new_s1, 0), torch.stack(new_s2, 0))
    return batch, new_n_beams, new_indices, new_removed_indices


def get_rawr(model, batch, target=None, max_beam_size=5, conf_threshold=-1,
             p_not_h=False):
    '''
    Args:
        p_not_h: reduce premise instead of hypothesis
        conf_threshold: (lower) threshold on confidence
    '''
    if target is None:
        target = model(batch.premise, batch.hypothesis)
        target = torch.max(target, 1)[1].data.cpu().numpy()

    batch = Batch(batch.premise, batch.hypothesis, batch.label)
    n_examples = batch.hypothesis.shape[0]
    n_beams = [1 for _ in range(n_examples)]

    # s2 is being reduced
    if p_not_h:
        s1, s2 = batch.hypothesis, batch.premise
    else:
        s1, s2 = batch.premise, batch.hypothesis

    indices = [list(range(real_length(x))) for x in s2]
    removed_indices = [[] for _ in range(n_examples)]

    final_s2 = [z for z in s2.data.cpu().numpy()]
    final_s2 = [[[int(x) for x in z if x != 1]] for z in final_s2]
    final_removed = [[[]] for _ in range(n_examples)]
    final_length = [len(x) for x in indices]

    while True:
        max_beam_size = min(s2.shape[1], max_beam_size)
        batch, n_beams, indices, removed_indices = remove_one(
                model, batch, n_beams,  indices,
                removed_indices, max_beam_size, p_not_h=p_not_h)
        model.eval()
        output = F.softmax(model(batch.premise, batch.hypothesis), 1)
        scores, preds = torch.max(output, 1)
        scores = scores.data.cpu().numpy()
        preds = preds.data.cpu().numpy()

        if p_not_h:
            s1, s2 = batch.hypothesis, batch.premise
        else:
            s1, s2 = batch.premise, batch.hypothesis

        start = 0
        new_s1, new_s2, new_indices, new_removed = [], [], [], []
        # batch, length
        for example_idx in range(n_examples):
            beam_size = 0
            for i in range(start, start + n_beams[example_idx]):
                if preds[i] == target[example_idx] \
                        and scores[i] >= conf_threshold:
                    new_length = real_length(s2[i])
                    ns2 = s2[i].data.cpu().numpy()
                    ns2 = [int(x) for x in ns2 if x != 1]
                    if new_length == final_length[example_idx]:
                        if ns2 not in final_s2[example_idx]:
                            final_s2[example_idx].append(ns2)
                            final_removed[example_idx].append(
                                    removed_indices[i])
                    elif new_length < final_length[example_idx]:
                        final_s2[example_idx] = [ns2]
                        final_removed[example_idx] = [removed_indices[i]]
                        final_length[example_idx] = new_length
                    if new_length == 1:
                        # this whols beam is length 1
                        # do not expand this beam
                        beam_size = 0
                        # break
                    else:
                        beam_size += 1
                        new_s2.append(s2[i])
                        new_s1.append(s1[i])
                        new_indices.append(indices[i])
                        new_removed.append(removed_indices[i])
            start += n_beams[example_idx]
            n_beams[example_idx] = beam_size

        if len(new_s2) == 0:
            # nothing to expand
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


def to_text(x, vocab):
    if isinstance(x, Variable):
        x = x.data
    if isinstance(x, torch.cuda.LongTensor):
        x = x.cpu()
    if isinstance(x, torch.LongTensor):
        x = x.numpy().tolist()
    return ' '.join(vocab[w] for w in x if w != 1)


def main():
    from args import conf, rawr_conf

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
    model.cuda(conf.gpu)

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
        target_scores, target = torch.max(output, 1)
        target = target.data.cpu()
        target_scores = target_scores.data.cpu()
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
                'prediction': target[i],
                'prediction_readable': a_vocab[target[i]],
                'score': target_scores[i],
                'label': batch_cpu.label[i],
                'label_readable': a_vocab[batch_cpu.label[i]]
                }
            checkpoint.append({'original': og, 'reduced': []})
            s1 = batch.hypothesis[i] if args.pnoth else batch.premise[i]
            if conf.gpu > -1:
                s1 = s1.cuda()
            for j, s2 in enumerate(reduced[i]):
                s2 = Variable(torch.LongTensor(s2))
                if conf.gpu > -1:
                    s2 = s2.cuda()
                model.eval()
                if args.pnoth:
                    output = model(s2.unsqueeze(0), s1.unsqueeze(0))
                else:
                    output = model(s1.unsqueeze(0), s2.unsqueeze(0))
                output = F.softmax(output, 1)
                pred_scores, pred = torch.max(output, 1)
                pred = pred.data[0]
                pred_scores = pred_scores.data[0]
                if args.pnoth:
                    hypo, prem = s1.data.cpu(), s2.data.cpu()
                else:
                    prem, hypo = s1.data.cpu(), s2.data.cpu()
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
            out_path = os.path.join(
                    out_dir, '{}.{}'.format(fname, batch_i))
            with open(out_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            checkpoint = []

    if len(checkpoint) > 0:
        out_path = os.path.join(
                out_dir, '{}.{}'.format(fname, batch_i))
        with open(out_path, 'wb') as f:
            pickle.dump(checkpoint, f)


if __name__ == '__main__':
    main()
