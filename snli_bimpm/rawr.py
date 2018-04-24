import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.BIMPM import BIMPM
from model.utils import SNLI, Quora
from util import prepare_output_dir


class Batch:
    
    def __init__(self, premise=None, hypothesis=None, label=None):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label = None
        
def real_length(x):
    # length of vector without padding
    if isinstance(x, Variable):
        return sum(x.data.cpu().numpy() != 1)
    else:
        return sum(x.cpu().numpy() != 1)

def get_onehot_grad(model, batch):
    criterion = nn.CrossEntropyLoss()
    extracted_grads = {}
    
    def extract_grad_hook(name):
        def hook(grad):
            extracted_grads[name] = grad
        return hook
    
    batch_size, length = batch.hypothesis.shape
    batch.premise.volatile = False
    batch.hypothesis.volatile = False
    model.eval()
    output = model(batch.premise, batch.hypothesis, embed_grad_hook=extract_grad_hook('embed'))
    label = torch.max(output, 1)[1]
    loss = criterion(output, label)
    loss.backward()
    embed_grad = extracted_grads['embed']
    embed = model.word_emb(batch.hypothesis)
    onehot_grad = embed.view(-1) * embed_grad.contiguous().view(-1)
    onehot_grad = onehot_grad.view(batch_size, length, -1).sum(-1)
    return onehot_grad

def remove_one(model, batch, n_beams, indices, removed_indices, max_beam_size):
    n_examples = len(n_beams) # not batch size!
    onehot_grad = get_onehot_grad(model, batch).data.cpu().numpy()
        
    start = 0
    new_hypo = []
    new_prem = []
    new_n_beams = []
    new_indices = []
    new_removed_indices = []
    hypo_lengths = [real_length(x) for x in batch.hypothesis]
    
    for example_idx in range(n_examples):
        if n_beams[example_idx] == 0:
            new_n_beams.append(0)
            continue
        
        coordinates = []
        for i in range(start, start + n_beams[example_idx]):
            if hypo_lengths[i] <= 1:
                continue
            order = np.argsort(- onehot_grad[i][:hypo_lengths[i]])
            coordinates += [(i, j) for j in order[:max_beam_size]]
            
        if len(coordinates) == 0:
            new_n_beams.append(0)
            start += n_beams[example_idx]
            continue
        
        coordinates = np.asarray(coordinates)
        scores = onehot_grad[coordinates[:, 0], coordinates[:, 1]]
        scores = sorted(zip(coordinates, scores), key=lambda x: -x[1])
        coordinates = [x for x, _ in scores[:max_beam_size]]
        
        assert all(j < hypo_lengths[i] for i, j in coordinates)

        cnt = 0
        for i, j in coordinates:
            partial_h = []
            if j > 0:
                partial_h.append(batch.hypothesis[i][:j])
            if j + 1 < batch.hypothesis[i].shape[0]:
                partial_h.append(batch.hypothesis[i][j + 1:])
            if len(partial_h) > 0:
                new_hypo.append(torch.cat(partial_h, 0))
                new_prem.append(batch.premise[i])
                new_removed_indices.append(removed_indices[i] + [indices[i][j]])
                new_indices.append(indices[i][:j] + indices[i][j+1:])
                cnt += 1
        new_n_beams.append(cnt)
        start += n_beams[example_idx]
    
    batch = Batch(torch.stack(new_prem, 0), torch.stack(new_hypo, 0))
    return batch, new_n_beams, new_indices, new_removed_indices

def get_rawr(model, batch, target=None, max_beam_size=5):
    target = torch.max(model(batch.premise, batch.hypothesis), 1)[1].data.cpu().numpy()
    
    batch = Batch(batch.premise, batch.hypothesis, batch.label)
    n_examples, _ = batch.hypothesis.shape
    n_beams = [1 for _ in range(n_examples)] # current number of beams for each example
    indices = [list(range(real_length(x))) for x in batch.hypothesis]
    removed_indices = [[] for _ in range(n_examples)]
    
    final_hypothesis = [z for z in batch.hypothesis.data.cpu().numpy()]
    final_hypothesis = [[[int(x) for x in z if x != 1]] for z in final_hypothesis]
    final_removed = [[[]] for _ in range(n_examples)]
    final_length = [len(x) for x in indices]
    
    while True:
        max_beam_size = min(batch.hypothesis.shape[1], 5)
        batch, n_beams, indices, removed_indices = remove_one(
                model, batch, n_beams,  indices, removed_indices, max_beam_size)
        model.eval()
        prediction = torch.max(model(batch.premise, batch.hypothesis), 1)[1].data.cpu().numpy()
        
        start = 0
        new_hypo, new_prem, new_indices, new_removed = [], [], [], []
        # batch, length
        for example_idx in range(n_examples):
            beam_size = 0
            for i in range(start, start + n_beams[example_idx]):
                if prediction[i] == target[example_idx]:
                    new_length = real_length(batch.hypothesis[i])
                    hypo = [int(x) for x in batch.hypothesis[i].data.cpu().numpy() if x != 1]
                    if new_length == final_length[example_idx]:
                        if hypo not in final_hypothesis[example_idx]:
                            final_hypothesis[example_idx].append(hypo)
                            final_removed[example_idx].append(removed_indices[i])
                    elif new_length < final_length[example_idx]:
                        final_hypothesis[example_idx] = [hypo]
                        final_removed[example_idx] = [removed_indices[i]]
                        final_length[example_idx] = new_length
                    if new_length == 1:
                        beam_size = 0
                        break
                    else:
                        beam_size += 1
                        new_hypo.append(batch.hypothesis[i])
                        new_prem.append(batch.premise[i])
                        new_indices.append(indices[i])
                        new_removed.append(removed_indices[i])
            start += n_beams[example_idx]
            n_beams[example_idx] = beam_size
        
        if len(new_hypo) == 0:
            break
        
        batch = Batch(torch.stack(new_prem, 0), torch.stack(new_hypo, 0))
        indices = new_indices
        removed_indices = new_removed
    return final_hypothesis, final_removed



def main():

    def to_text(x):
        if isinstance(x, Variable):
            x = x.data
        if isinstance(x, torch.cuda.LongTensor):
            x = x.cpu()
        if isinstance(x, torch.LongTensor):
            x = x.numpy().tolist()
        return ' '.join(q_vocab[w] for w in x if w != 1)

    from args import args

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--model', required=True)
    _args = parser.parse_args()
    args.fold = _args.fold
    args.model_path = _args.model
    # args.model_path = 'saved_models/BIBPM_SNLI_16:48:18.pt'

    data = SNLI(args)

    args.char_vocab_size = len(data.char_vocab)
    args.word_vocab_size = len(data.TEXT.vocab)
    args.class_size = len(data.LABEL.vocab)
    args.max_word_len = data.max_word_len

    q_vocab = data.TEXT.vocab.itos
    a_vocab = data.LABEL.vocab.itos

    out_dir = prepare_output_dir(args, args.root_dir)
    print('Generating [{}] rawr data from [{}].'.format(args.fold, args.model_path))
    print(out_dir)
    
    model = BIMPM(args, data)
    model.load_state_dict(torch.load(args.model_path))
    model.word_emb.weight.requires_grad = True
    model.cuda(args.gpu)

    datasets = {'train': data.train_iter, 'dev': data.dev_iter}

    checkpoint = []
    for batch_i, batch in enumerate(tqdm(datasets[args.fold])):
        # if batch_i > len(datasets[args.fold]):
        #     break
        # if batch_i > 2500:
        #     break
        # if batch_i <= 7500:
        #     continue
        # if batch_i > 10000:
        #     break

        n_examples = batch.hypothesis.shape[0]

        model.eval()
        output = F.softmax(model(batch.premise, batch.hypothesis), 1)
        target_scores, target = torch.max(output, 1)
        target = target.data.cpu().numpy()
        target_scores = target_scores.data.cpu().numpy()

        reduced, removed_indices = get_rawr(model, batch, max_beam_size=5)
        model.eval()
        for i in range(n_examples):
            checkpoint.append([])
            for j, hypo in enumerate(reduced[i]):
                if args.fold == 'dev':
                    test_prem = batch.premise[i].unsqueeze(0)
                    test_hypo = Variable(torch.LongTensor([hypo])).cuda()
                    output = F.softmax(model(test_prem, test_hypo), 1)
                    pred_scores, pred = torch.max(output, 1)
                    pred = pred.data.cpu().numpy()[0]
                    pred_scores = pred_scores.data.cpu().numpy()[0]
                
                    checkpoint[-1].append({
                        'data': {
                            'premise': test_prem.data.cpu(),
                            'hypothesis': test_hypo.data.cpu(),
                            'label': batch.label[i].cpu()
                            },
                        'premise': to_text(batch.premise[i]),
                        'original_hypothesis': to_text(batch.hypothesis[i]),
                        'reduced_hypothesis': to_text(hypo),
                        'original_prediction': a_vocab[target[i]],
                        'original_score': target_scores[i],
                        'reduced_prediction': a_vocab[pred],
                        'reduced_score': pred_scores,
                        'removed_indices': removed_indices[i][j],
                        'label': a_vocab[batch.label.data.cpu().numpy()[i]]
                        })
                else:
                    test_prem = batch.premise[i].unsqueeze(0)
                    test_hypo = torch.LongTensor([hypo])
                    checkpoint[-1].append({
                        'data': {
                            'premise': test_prem.data.cpu(),
                            'hypothesis': test_hypo.cpu(),
                            'label': batch.label[i].cpu()
                            },
                        })
    out_path = os.path.join(out_dir, 'rawr.{}.pkl'.format(args.fold))
    with open(out_path, 'wb') as f:
        pickle.dump(checkpoint, f)

if __name__ == '__main__':
    main()
