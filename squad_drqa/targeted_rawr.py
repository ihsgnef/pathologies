import re
import os
import sys
import json
import random
import string
import pickle
import msgpack
import logging
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from shutil import copyfile
from datetime import datetime
from collections import Counter

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from drqa.model import DocReaderModel
from drqa.utils import str2bool
from util import BatchGen, load_data, score
from util import prepare_output_dir
from rawr import remove_one, real_length

def get_targeted_rawr(model, batch, target_answer, target_s, target_e, max_beam_size=5):
    n_examples = batch[5].shape[0]
    n_beams = [1 for _ in range(n_examples)] # current number of beams for each example
    # reduced_question = [x.cpu().numpy().tolist() for x in batch[5]]
    # reduced_question = [[[int(w) for w in x if w != 0]] for x in reduced_question]
    reduced_question = [[] for x in batch[5]]
    reduced_length = [real_length(x) for x in batch[5]]
    indices = [list(range(real_length(x))) for x in batch[5]] # remaining indices
    removed_indices = [[] for _ in range(n_examples)] # removed indices
    final_removed_indices = [[[]] for _ in range(n_examples)]
    
    while True:
        max_beam_size = min(batch[5].shape[1], max_beam_size)
        batch, n_beams, indices, removed_indices, target_s, target_e = remove_one(
            model, batch, n_beams, indices, removed_indices, max_beam_size,
            target_s=target_s, target_e=target_e)
        prediction = model.predict(batch)
        
        start = 0
        new_target_s, new_target_e = [], []
        new_batch, new_indices, new_removed_indices = [], [], []
        for example_idx in range(n_examples):
            beam_size = 0
            for i in range(start, start + n_beams[example_idx]):
                new_length = real_length(batch[5][i])
                if prediction[i] == target_answer[example_idx]: 
                    b5 = batch[5][i].cpu().numpy().tolist()
                    b5 = [int(w) for w in b5 if w != 0] # remove padding
                    if new_length == reduced_length[example_idx]:
                        if b5 not in reduced_question[example_idx]:
                            reduced_question[example_idx].append(b5)
                            final_removed_indices[example_idx].append(removed_indices[i])
                    if new_length < reduced_length[example_idx]:
                        reduced_length[example_idx] = new_length
                        reduced_question[example_idx] = [b5]
                        final_removed_indices[example_idx] = [removed_indices[i]]
                if new_length == 1:
                    # FIXME stop if there is only one word left
                    # because current model cannot handle that
                    continue
                else:
                    beam_size += 1
                    new_batch.append([x[i] for x in batch])
                    new_indices.append(indices[i])
                    new_removed_indices.append(removed_indices[i])
                    new_target_s.append(target_s[i])
                    new_target_e.append(target_e[i])
            start += n_beams[example_idx]
            n_beams[example_idx] = beam_size
            
        if len(new_batch) == 0:
            break
        
        new_batch = list(map(list, zip(*new_batch)))
        batch = [torch.stack(c, 0) for c in new_batch[:7]]
        batch += new_batch[7:]
        target_s = torch.cat(new_target_s)
        target_e = torch.cat(new_target_e)
        indices = new_indices
        removed_indices = new_removed_indices
    return reduced_question, final_removed_indices

def main_targeted():
    from args import args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--fold', required=True)
    _args = parser.parse_args()
    args.load_model_dir = _args.model
    args.fold = _args.fold
    out_dir = prepare_output_dir(args, 'results')
    print('Generating [{}] targeted rawr data from [{}].'.format(args.fold, args.load_model_dir))
    pkl_dir = os.path.join(out_dir, '{}.targeted.pkl'.format(args.fold))
    print('Saving to {}'.format(pkl_dir))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    train, dev, dev_y, embedding, opt = load_data(args)
    data = {'train': train, 'dev': dev}

    state_dict = torch.load(args.load_model_dir)['state_dict']
    model = DocReaderModel(vars(opt), embedding, state_dict)
    model.cuda()

    batches = {}
    batches['train'] = BatchGen(
            [x[:8] for x in train], batch_size=30,
            pos_size=args.pos_size, ner_size=args.ner_size,
            gpu=args.cuda, evaluation=True)
    batches['dev'] = BatchGen(dev, batch_size=30,
            pos_size=args.pos_size, ner_size=args.ner_size,
            gpu=args.cuda, evaluation=True)

    all_reduced = []
    all_removed = []
    example_idx = 0
    for batch_i, batch in enumerate(tqdm(batches[args.fold])):
        # if batch_i > 10:
        #     break
        n_examples = batch[1].shape[0]
        answers, _, score_s, score_e, _, _ = model.predict(batch, get_all=True)
        target_s = Variable(torch.max(score_s, 1)[1]).cuda()
        target_e = Variable(torch.max(score_e, 1)[1]).cuda()
        reduced, removed = get_targeted_rawr(
                model, batch, answers, target_s, target_e, max_beam_size=5)
        for i in range(n_examples):
            idx = example_idx + i
            assert batch[8][i] == data[args.fold][idx][7] # check if the spans match
            all_reduced.append([])
            for j, e in enumerate(reduced[i]):
                x = list(data[args.fold][idx])
                x[5] = e 
                all_reduced[-1].append(x)
            all_removed.append(removed[i])
        example_idx += n_examples

    with open(pkl_dir, 'wb') as f:
        ckp = {'reduced': all_reduced, 'removed': all_removed}
        pickle.dump(ckp, f)

if __name__ == '__main__':
    main_targeted()
