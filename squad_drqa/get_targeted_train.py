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
from collections import defaultdict

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from drqa.model import DocReaderModel
from drqa.utils import str2bool
from util import BatchGen, load_data, score
from util import prepare_output_dir
from targeted_rawr import get_targeted_rawr

def main():
    from args import args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    _args = parser.parse_args()
    args.load_model_dir = _args.model

    out_dir = prepare_output_dir(args, 'results')
    print('Generating targeted rawr data from [{}].'.format(args.load_model_dir))
    print('Saving to {}'.format(out_dir))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    train, dev, dev_y, embedding, opt = load_data(args)
    data = {'train': train, 'dev': dev}

    state_dict = torch.load(args.load_model_dir)['state_dict']
    model = DocReaderModel(vars(opt), embedding, state_dict)
    model.cuda()

    # get answers and targets for original question
    regular_train = BatchGen(
            [x[:8] for x in train], batch_size=30,
            pos_size=args.pos_size, ner_size=args.ner_size,
            gpu=args.cuda, evaluation=True)
    
    all_answers, all_target_s, all_target_e = [], [], []
    for i, batch in enumerate(tqdm(regular_train)):
        # if i > 10:
        #     break
        answers, _, score_s, score_e, _, _ = model.predict(batch, get_all=True)
        target_s = np.argmax(score_s, 1).tolist()
        target_e = np.argmax(score_e, 1).tolist()
        all_answers += answers
        all_target_s += target_s
        all_target_e += target_e

    all_train = zip(train, all_answers, all_target_s, all_target_s)

    groups = defaultdict(list)
    for x in all_train:
        groups[x[0][0][:-2]].append(x)

    train_with_other = []
    other_answers = []
    other_target_s = []
    other_target_e = []
    for group in groups.values():
        data, answer, target_s, target_e = list(map(list, zip(*group)))
        for i in range(len(data)):
            for j in range(len(data)):
                if answer[i] != answer[j]:
                    train_with_other.append(data[i])
                    other_answers.append(answer[j])
                    other_target_s.append(target_s[j])
                    other_target_e.append(target_e[j])

    other_train = BatchGen(
            [x[:8] for x in train_with_other], batch_size=30,
            pos_size=args.pos_size, ner_size=args.ner_size,
            gpu=args.cuda, evaluation=True)
    
    targeted_train = []
    start = 0
    for i, batch in enumerate(tqdm(other_train)):
        batch_size = batch[5].shape[0]
        end = start + batch_size
        # if i >= 2500:
        #     break

        # if i < 2500 or i == 2530:
        #     start = end
        #     continue
        # if i >=5000:
        #     break
        
        # if i < 5000:
        #     start = end
        #     continue
        # if i >= 7500:
        #     break

        if i < 7500:
            start = end
            continue

        ts = Variable(torch.LongTensor(other_target_s[start : end])).cuda()
        te = Variable(torch.LongTensor(other_target_e[start : end])).cuda()
        ans = other_answers[start : end]
        reduced, _ = get_targeted_rawr(model, batch, ans, ts, te)
        for j in range(batch_size):
            if len(reduced[j]) == 0:
                continue
            for r in reduced[j]:
                x = train_with_other[start + j]
                x[5] = r
                targeted_train.append(x)
        start = end

    print(len(targeted_train))

    out_path = os.path.join(out_dir, 'targeted_other.train.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(targeted_train, f)

if __name__ == '__main__':
    main()
