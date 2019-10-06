import os
import random
import pickle
import msgpack
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from drqa.model import DocReaderModel
from util import BatchGen, load_data
from util import prepare_output_dir


def get_onehot_grad(model, batch, target_s=None, target_e=None):
    extracted_grads = {}

    def get_grad_hook(name):
        def hook(grad):
            extracted_grads[name] = grad
        return hook

    batch_size, length = batch[5].shape
    model.network.eval()
    inputs = [Variable(e).cuda() for e in batch[:7]
              if not isinstance(e, Variable)]
    score_s, score_e = model.network(*inputs, grad_hook=get_grad_hook('embed'))
    embed = model.network.embedding(inputs[5])
    if target_s is None:
        target_s = torch.max(score_s, 1)[1]
    if target_e is None:
        target_e = torch.max(score_e, 1)[1]
    loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
    model.network.zero_grad()
    loss.backward()
    embed_grad = extracted_grads['embed']
    onehot_grad = embed.view(-1) * embed_grad.view(-1)
    onehot_grad = onehot_grad.view(batch_size, length, -1).sum(-1)
    return onehot_grad


def remove_one(model, batch, n_beams, indices, removed_indices, max_beam_size,
               target_s=None, target_e=None):
    n_examples = len(n_beams)
    has_label = False
    if target_s is not None and target_e is not None:
        has_label = True
        onehot_grad = get_onehot_grad(model, batch, target_s, target_e)
    else:
        onehot_grad = get_onehot_grad(model, batch)

    onehot_grad = onehot_grad.data.cpu().numpy()
    question = batch[5]
    question_mask = batch[6]
    question_lengths = [real_length(x) for x in question]

    new_batch = []
    new_n_beams = []
    new_indices = []
    new_removed_indices = []
    if has_label:
        new_target_s = []
        new_target_e = []

    start = 0
    for example_idx in range(n_examples):
        if n_beams[example_idx] == 0:
            new_n_beams.append(0)
            continue

        coordinates = []  # i_in_batch, j_in_question
        # ignore PADs
        for i in range(start, start + n_beams[example_idx]):
            if real_length(question[i]) == 1:
                continue
            word_order = np.argsort(-onehot_grad[i][:question_lengths[i]])
            coordinates += [(i, j) for j in word_order[:max_beam_size]]

        if len(coordinates) == 0:
            start += n_beams[example_idx]
            new_n_beams.append(0)
            continue

        coordinates = np.asarray(coordinates)
        scores = onehot_grad[coordinates[:, 0], coordinates[:, 1]]
        scores = sorted(zip(coordinates, scores), key=lambda x: -x[1])
        coordinates = np.asarray([x for x, _ in scores[:max_beam_size]])
        assert all(j < question_lengths[i] for i, j in coordinates)
        if not all(j < len(indices[i]) for i, j in coordinates):
            for i, j in coordinates:
                print('i', i)
                print('j', j)
                print('ql', question_lengths[i])
                print('len(indices)', len(indices[i]))
                print(indices[i])
                print()

        cnt = 0
        for i, j in coordinates:
            # because stupid tensor doesn't support proper indexing
            q, qm = [], []
            if j > 0:
                q.append(question[i][:j])
                qm.append(question_mask[i][:j])
            if j + 1 < question[i].shape[0]:
                q.append(question[i][j + 1:])
                qm.append(question_mask[i][j + 1:])
            if len(q) > 0:
                new_entry = [x[i] for x in batch]
                new_entry[5] = torch.cat(q, 0)
                new_entry[6] = torch.cat(qm, 0)
                new_batch.append(new_entry)
                new_removed_indices.append(removed_indices[i] + [indices[i][j]])
                new_indices.append(indices[i][:j] + indices[i][j+1:])
                if has_label:
                    new_target_s.append(target_s[i])
                    new_target_e.append(target_e[i])
                cnt += 1
        start += n_beams[example_idx]
        new_n_beams.append(cnt)

    new_batch = list(map(list, zip(*new_batch)))
    batch = [torch.stack(c, 0) for c in new_batch[:7]]
    batch += new_batch[7:]
    if has_label:
        new_target_s = torch.cat(new_target_s)
        new_target_e = torch.cat(new_target_e)
        return batch, new_n_beams, new_indices, new_removed_indices, \
            new_target_s, new_target_e
    else:
        return batch, new_n_beams, new_indices, new_removed_indices


def real_length(x):
    # length of vector without padding
    if isinstance(x, np.ndarray):
        return sum(x != 0)
    if isinstance(x, Variable):
        return sum(x.data.cpu().numpy() != 0)
    else:
        return sum(x.cpu().numpy() != 0)


def get_rawr(model, batch, max_beam_size=5):
    target = model.predict(batch)
    n_examples = batch[5].shape[0]
    n_beams = [1 for _ in range(n_examples)]
    reduced_question = [x.cpu().numpy().tolist() for x in batch[5]]
    reduced_question = [[[int(w) for w in x if w != 0]] for x in reduced_question]
    reduced_length = [real_length(x) for x in batch[5]]
    indices = [list(range(real_length(x))) for x in batch[5]]
    removed_indices = [[] for _ in range(n_examples)]
    final_removed_indices = [[[]] for _ in range(n_examples)]

    while True:
        max_beam_size = min(batch[5].shape[1] - 1, max_beam_size)
        batch, n_beams, indices, removed_indices = remove_one(
            model, batch, n_beams, indices, removed_indices, max_beam_size)
        prediction = model.predict(batch)

        start = 0
        new_batch, new_indices, new_removed_indices = [], [], []
        for example_idx in range(n_examples):
            beam_size = 0
            for i in range(start, start + n_beams[example_idx]):
                if prediction[i] == target[example_idx]:
                    new_length = real_length(batch[5][i])
                    b5 = batch[5][i].cpu().numpy().tolist()
                    b5 = [int(w) for w in b5 if w != 0]
                    if new_length == reduced_length[example_idx]:
                        if b5 not in reduced_question[example_idx]:
                            reduced_question[example_idx].append(b5)
                            final_removed_indices[example_idx].append(removed_indices[i])
                    if new_length < reduced_length[example_idx]:
                        reduced_length[example_idx] = new_length
                        reduced_question[example_idx] = [b5]
                        final_removed_indices[example_idx] = [removed_indices[i]]
                    if new_length == 1:
                        # all entries in this beam have length 1
                        # do not expand this beam
                        beam_size = 0
                    else:
                        beam_size += 1
                        new_batch.append([x[i] for x in batch])
                        new_indices.append(indices[i])
                        new_removed_indices.append(removed_indices[i])
            start += n_beams[example_idx]
            n_beams[example_idx] = beam_size

        if len(new_batch) == 0:
            break

        new_batch = list(map(list, zip(*new_batch)))
        batch = [torch.stack(c, 0) for c in new_batch[:7]]
        batch += new_batch[7:]
        indices, removed_indices = new_indices, new_removed_indices
    return reduced_question, final_removed_indices


def batch_repeat(arr, batch_size):
    # repeat array and form a batch
    if isinstance(arr, list) or isinstance(arr, str):
        return [arr for _ in range(batch_size)]
    else:
        shape = [batch_size] + [1 for _ in range(arr.ndimension())]
        return arr.unsqueeze(0).repeat(*shape)


def main():
    from args import conf, rawr_conf

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='results/baseline.pt')
    parser.add_argument('--fold', required=True)
    args = parser.parse_args()
    out_dir = prepare_output_dir(conf, 'results', 'rawr')

    pkl_dir = os.path.join(out_dir, '{}.pkl'.format(args.fold))
    print('Generating [{}] rawr data from [{}].'.format(args.fold, args.model))
    print('Saving to {}'.format(pkl_dir))

    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)

    with open('data/meta.msgpack', 'rb') as f:
        vocab = msgpack.load(f, encoding='utf-8')['vocab']

    train, dev, dev_y, embedding, opt = load_data(conf)
    data = {'train': train, 'dev': dev}

    state_dict = torch.load(args.model)['state_dict']
    model = DocReaderModel(vars(opt), embedding, state_dict)
    model.cuda()

    batches = {}
    batches['train'] = BatchGen(
            [x[:8] for x in train], batch_size=30,
            pos_size=conf.pos_size, ner_size=conf.ner_size,
            gpu=conf.cuda, evaluation=True)
    batches['dev'] = BatchGen(
            dev, batch_size=30,
            pos_size=conf.pos_size, ner_size=conf.ner_size,
            gpu=conf.cuda, evaluation=True)

    checkpoint = []
    example_idx = 0
    for batch_i, batch in enumerate(tqdm(batches[args.fold])):
        n_examples = batch[1].shape[0]

        # original predictions
        r = model.predict(batch, get_all=True)
        target = r[0]
        original_score_s = r[2].cpu().numpy()
        original_score_e = r[3].cpu().numpy()
        original_index_s = r[4]
        original_index_e = r[5]

        reduced, removed = get_rawr(model, batch,
                                    max_beam_size=rawr_conf.max_beam_size)

        for i in range(n_examples):
            beam_size = len(reduced[i])

            rq = torch.LongTensor(reduced[i])
            mask = torch.ByteTensor(np.zeros_like(rq))
            test_batch = [batch_repeat(x[i], beam_size) for x in batch[:5]]
            test_batch += [rq, mask]
            test_batch += [batch_repeat(x[i], beam_size) for x in batch[7:]]

            output = model.predict(test_batch, get_all=True)
            preds = output[0]
            reduced_score_s = output[2].cpu().numpy()
            reduced_score_e = output[3].cpu().numpy()
            reduced_index_s = output[4]
            reduced_index_e = output[5]

            idx = example_idx + i
            assert batch[8][i] == data[args.fold][idx][7]  # check if the spans match

            if args.fold == 'train':
                indices = data['train'][idx][i][7]
                start, end = data['train'][i][-2], data['train'][i][-1]
                start, end = indices[start][0], indices[end][1]
                label = [train[i][6][start:end]]
            else:
                label = dev_y[idx]

            og = {
                'batch': data[args.fold][idx],
                'label': label,
                'score_e': original_score_e[i],
                'score_s': original_score_s[i],
                'index_e': original_index_e[i],
                'index_s': original_index_s[i],
                'prediction': target[i],
                'context_readable': batch[7][i],
                'question_readable': ' '.join(vocab[x]
                                              for x in batch[5][i]
                                              if x != 0),
                }

            rd = []
            for j, e in enumerate(reduced[i]):
                x = list(data[args.fold][idx])
                x[5] = e
                rd.append({
                    'batch': x,
                    'label': label,
                    'removed_indices': removed[i][j],
                    'context_readable': batch[7][i],
                    'question_readable': ' '.join(vocab[x]
                                                  for x in reduced[i][j]
                                                  if x != 0),
                    'score_e': reduced_score_e[j],
                    'score_s': reduced_score_s[j],
                    'index_e': reduced_index_e[j],
                    'index_s': reduced_index_s[j],
                    'prediction': preds[j]
                })

            checkpoint.append({'original': og, 'reduced': rd})

        example_idx += n_examples

    with open(pkl_dir, 'wb') as f:
        pickle.dump(checkpoint, f)


if __name__ == '__main__':
    main()
