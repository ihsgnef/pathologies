import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import data
import model
import config
from utils import prepare_output_dir

def get_onehot_grad(model, batch):
    extracted_grads = {}
    
    def get_grad_hook(name):
        def hook(grad):
            extracted_grads[name] = grad
        return hook

    model.eval()
    v, q, a, idx, q_len = batch
    batch_size, length = q.shape
    v = Variable(v).cuda()
    q = Variable(q).cuda()
    q_len = Variable(q_len).cuda()
    
    out = model(v, q, q_len, embed_grad_hook=get_grad_hook('embed'))
    embed = model.module.text.embedding(q)
    pred = torch.max(out, 1)[1]
    loss = F.nll_loss(F.log_softmax(out, 1), pred)
    model.zero_grad()
    loss.backward()
    embed_grad = extracted_grads['embed']
    onehot_grad = embed.view(-1) * embed_grad.view(-1)
    onehot_grad = onehot_grad.view(batch_size, length, -1).sum(-1)
    return onehot_grad

def real_length(x):
    # length of vector without padding
    if isinstance(x, np.ndarray):
        return sum(x != 0)
    if isinstance(x, Variable):
        return sum(x.data.cpu().numpy() != 0)
    else:
        return sum(x.cpu().numpy() != 0)

def remove_one(model, batch, n_beams, indices, removed_indices, max_beam_size):
    n_examples = len(n_beams)
    onehot_grad = get_onehot_grad(model, batch)
    onehot_grad = onehot_grad.data.cpu().numpy()
    
    v, q, a, idx, q_len = batch

    new_batch = []
    new_n_beams = []
    new_indices = []
    new_removed_indices = []

    start = 0
    for example_idx in range(n_examples):
        if n_beams[example_idx] == 0:
            new_n_beams.append(0)
            continue
        
        coordinates = [] # i_in_batch, j_in_question
        # ignore PADs
        for i in range(start, start + n_beams[example_idx]):
            if q_len[i] == 1:
                continue
            word_order = np.argsort(-onehot_grad[i][:q_len[i]])
            coordinates += [(i, j) for j in word_order[:max_beam_size]]

        if len(coordinates) == 0:
            start += n_beams[example_idx]
            new_n_beams.append(0)
            continue
            
        coordinates = np.asarray(coordinates)
        scores = onehot_grad[coordinates[:, 0], coordinates[:, 1]]
        scores = sorted(zip(coordinates, scores), key=lambda x: -x[1])
        coordinates = np.asarray([x for x, _ in scores[:max_beam_size]])
        assert all(j < q_len[i] for i, j in coordinates)
        
        cnt = 0
        for i, j in coordinates:
            partial_q = []
            if j > 0:
                partial_q.append(q[i][:j])
            if j + 1 < q[i].shape[0]:
                partial_q.append(q[i][j + 1:])
            if len(partial_q) > 0:
                new_entry = [v[i], None, a[i], idx[i], q_len[i] - 1]
                new_entry[1] = torch.cat(partial_q, 0)
                new_batch.append(new_entry)
                new_removed_indices.append(removed_indices[i] + [indices[i][j]])
                new_indices.append(indices[i][:j] + indices[i][j+1:])
                cnt += 1
        start += n_beams[example_idx]
        new_n_beams.append(cnt)
        
    v, q, a, idx, q_len = list(map(list, zip(*new_batch)))
    v = torch.stack(v, 0)
    q = torch.stack(q, 0)
    a = torch.stack(a, 0)
    q_len = torch.LongTensor(q_len).cuda()
    batch = [v, q, a, idx, q_len]
    return batch, new_n_beams, new_indices, new_removed_indices

def get_rawr(model, batch, max_beam_size=5):
    v, q, a, idx, q_len = batch
    inputs = [Variable(e).cuda() for e in [v, q, q_len]]
    model.eval()
    target = torch.max(model(*inputs), 1)[1].data.cpu().numpy()
    n_examples = q.shape[0]
    n_beams = [1 for _ in range(n_examples)]
    reduced_q = [x.cpu().numpy().tolist() for x in q]
    reduced_q = [[[int(w) for w in x if w != 0]] for x in reduced_q]
    reduced_length = q_len
    indices = [list(range(x)) for x in q_len]
    removed_indices = [[] for _ in range(n_examples)]
    final_removed_indices = [[[]] for _ in range(n_examples)]
    
    while True:
        max_beam_size = min(batch[1].shape[1], max_beam_size)
        batch, n_beams, indices, removed_indices = remove_one(
            model, batch, n_beams, indices, removed_indices, max_beam_size)
        v, q, a, idx, q_len = batch
                              
        inputs = [Variable(e).cuda() for e in [v, q, q_len] if not isinstance(e, Variable)]
        model.eval()
        predictions = torch.max(model(*inputs), 1)[1].data.cpu().numpy()
        
        start = 0
        new_batch, new_indices, new_removed_indices = [], [], []
        for example_idx in range(n_examples):
            beam_size = 0
            for i in range(start, start + n_beams[example_idx]):
                if predictions[i] == target[example_idx]: 
                    nq = q[i].cpu().numpy().tolist()
                    nq = [int(w) for w in nq if w != 0] # remove padding
                    if q_len[i] == reduced_length[example_idx]:
                        if nq not in reduced_q[example_idx]:
                            reduced_q[example_idx].append(nq)
                            final_removed_indices[example_idx].append(removed_indices[i])
                    if q_len[i] < reduced_length[example_idx]:
                        reduced_length[example_idx] = q_len[i]
                        reduced_q[example_idx] = [nq]
                        final_removed_indices[example_idx] = [removed_indices[i]]
                    if q_len[i] == 1:
                        # FIXME stop if there is only one word left
                        # because current model cannot handle that
                        continue
                    else:
                        beam_size += 1
                        new_batch.append([v[i], q[i], a[i], idx[i], q_len[i]])
                        new_indices.append(indices[i])
                        new_removed_indices.append(removed_indices[i])
            start += n_beams[example_idx]
            n_beams[example_idx] = beam_size
            
        if len(new_batch) == 0:
            break
        
        v, q, a, idx, q_len = list(map(list, zip(*new_batch)))
        v = torch.stack(v, 0)
        q = torch.stack(q, 0)
        a = torch.stack(a, 0)
        q_len = torch.LongTensor(q_len)
        batch = [v, q, a, idx, q_len]
        indices, removed_indices = new_indices, new_removed_indices
    return reduced_q, final_removed_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--fold', required=True)
    parser.add_argument('--model', default='2017-08-04_00.55.19.pth')
    _args = parser.parse_args()
    fold = _args.fold
    fast = _args.fast
    model_path = _args.model
    out_dir = prepare_output_dir(user_dir='results')
    print(out_dir, fold)

    ckp = torch.load(model_path)
    # vocab_size = len(ckp['vocab']['question']) + 1
    vocab_size = 15193
    net = nn.DataParallel(model.Net(vocab_size))
    net.load_state_dict(ckp['weights'])
    # net.load_state_dict(ckp)
    net.cuda()

    if fold == 'dev':
        data_iter = data.get_loader(val=True)
        coco = data.CocoImages(config.val_path)
    elif fold == 'train':
        data_iter = data.get_loader(train=True)
        coco = data.CocoImages(config.train_path)
    q_vocab = {v:k for k, v in data_iter.dataset.vocab['question'].items()}
    a_vocab = {v:k for k, v in data_iter.dataset.vocab['answer'].items()}

    checkpoint = []
    lens = []
    for batch_idx, batch in enumerate(tqdm(data_iter)):
        # if batch_idx > 10:
        #     break

        if batch_idx > len(data_iter):
            break

        v, q, a, idx, q_len = batch
        batch_size = v.shape[0]

        # original predictions
        net.eval()
        inputs = [Variable(e).cuda() for e in [v, q, q_len]]
        output = F.softmax(net(*inputs), 1)
        target_scores, target = torch.max(output, 1)
        target_scores = target_scores.data.cpu().numpy()
        target = target.data.cpu().numpy()

        reduced, removed_ids = get_rawr(net, batch, max_beam_size=5)
        for i in range(batch_size):
            if len(reduced[i][0]) == 0:
                reduced[i][0] = [0]
            if fast:
                lens.append(len(reduced[i][0]))
                coco_id = data_iter.dataset.coco_ids[idx[i]]
                coco_idx = coco.sorted_ids.index(coco_id)
                checkpoint.append({
                        'reduced': reduced[i],
                        'removed_indices': removed_ids[i],
                        'original_question': ' '.join(q_vocab[x] for x in q[i] if x != 0),
                        'reduced_question': ' '.join(q_vocab[x] for x in reduced[i][0] if x != 0),
                        'original_prediction': a_vocab[target[i]],
                        'coco_idx': coco_idx,
                        'idx': idx[i],
                    })
            else:
                test_batch = [
                        v[i].unsqueeze(0),
                        torch.LongTensor([reduced[i][0]]),
                        torch.LongTensor([len(reduced[i][0])])]
                test_batch = [Variable(e).cuda() for e in test_batch]
                net.eval()
                output = F.softmax(net(*test_batch), 1)
                pred_scores, pred = torch.max(output, 1)
                pred_scores = pred_scores.data.cpu().numpy()[0]
                pred = pred.data.cpu().numpy()[0]

                coco_id = data_iter.dataset.coco_ids[idx[i]]
                coco_idx = coco.sorted_ids.index(coco_id)

                labels = a[i].cpu().numpy()
                order = np.argsort(-labels)
                scores = labels[order].tolist()
                labels = [(a_vocab[o], s) for o, s in zip(order, scores) if s > 0]
                lens.append(len(reduced[i][0]))

                checkpoint.append({
                        'reduced': reduced[i],
                        'removed_indices': removed_ids[i],
                        'original_question': ' '.join(q_vocab[x] for x in q[i] if x != 0),
                        'reduced_question': ' '.join(q_vocab[x] for x in reduced[i][0] if x != 0),
                        'original_prediction': a_vocab[target[i]],
                        'reduced_prediction': a_vocab[pred],
                        'labels': labels,
                        'original_score': target_scores[i],
                        'reduced_score': pred_scores,
                        'coco_idx': coco_idx,
                        'idx': idx[i],
                    })

    with open(os.path.join(out_dir, 'rawr.{}.pkl'.format(fold)), 'wb') as f:
        pickle.dump(checkpoint, f)

    print(np.mean(lens))

if __name__ == '__main__':
    main()
