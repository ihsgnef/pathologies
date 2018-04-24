import os
import sys
import copy
import pickle
import random
import argparse
import logging
import scipy
import scipy.stats
import numpy as np
import itertools

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from time import gmtime, strftime

from model.BIMPM import BIMPM
from model.utils import SNLI, Quora
from test import test
from distribution import entropy
from util import prepare_output_dir

def forward(model, premise, hypothesis, args=None):
    if not isinstance(premise, Variable):
        premise = Variable(premise).cuda()
    if not isinstance(hypothesis, Variable):
        hypothesis = Variable(hypothesis).cuda()
    if args is not None:
        if args.max_sent_len >= 0:
            if premise.shape[1] > args.max_sent_len:
                premise = premise[:, :args.max_sent_len]
            if hypothesis.shape[1] > args.max_sent_len:
                hypothesis = hypothesis[:, :args.max_sent_len]
    output = model(premise, hypothesis)
    return output

def evaluate(model, dev_iter, args):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    acc, loss, size = 0, 0, 0
    for batch in dev_iter:
        pred = model(batch.premise, batch.hypothesis)
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]
        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)
    acc /= size
    acc = acc.cpu().data[0]
    return loss, acc

def evaluate_ent(model, dev_iter, args):
    model.eval()
    acc, ent, size = 0, 0, 0
    for premise, hypothesis, label in dev_iter:
        output = F.softmax(forward(model, premise, hypothesis), 1)
        ent += entropy(output).sum().data.cpu().numpy()[0]
        _, pred = output.max(dim=1)
        if isinstance(label, Variable):
            label = label.data.cpu()
        acc += (pred.data.cpu() == label).sum()
        size += hypothesis.shape[0]
    ent = ent / size
    acc /= size
    acc = acc
    return ent, acc


def batchify(data, batch_size):
    data = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    batches = []
    for batch in data:
        prem, hypo, label = list(map(list, zip(*batch)))
        prem_len = max(x.shape[1] for x in prem)
        for i, x in enumerate(prem):
            if x.shape[1] < prem_len:
                ones = np.ones((1, prem_len - x.shape[1]))
                ones = torch.LongTensor(ones)
                prem[i] = torch.cat([x, ones], 1)
        hypo_len = max(x.shape[1] for x in hypo)
        for i, x in enumerate(hypo):
            if x.shape[1] < hypo_len:
                ones = np.ones((1, hypo_len - x.shape[1]))
                ones = torch.LongTensor(ones)
                hypo[i] = torch.cat([x, ones], 1)
        prem = torch.cat(prem, 0)
        hypo = torch.cat(hypo, 0)
        label = torch.stack(label, 0)
        batches.append((prem, hypo, label))
    return batches


def main():
    from args import args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='saved_models/BIBPM_SNLI_16:48:18.pt')
    parser.add_argument('--n-mle', default=2, type=int)
    parser.add_argument('--n-ent', default=2, type=int)
    parser.add_argument('--ent-lr', default=2e-4)
    parser.add_argument('--ent-train', 
            default='/scratch0/shifeng/rawr/new_snli/rawr.train.pkl')
    parser.add_argument('--ent-dev',
            default='/scratch0/shifeng/rawr/new_snli/rawr.dev.pkl')
    parser.add_argument('--gamma', default=1e-3)
    new_args = parser.parse_args()
    args.model_path = new_args.model
    args.n_mle = new_args.n_mle
    args.n_ent = new_args.n_ent
    args.ent_lr = new_args.ent_lr
    args.ent_train = new_args.ent_train
    args.ent_dev = new_args.ent_dev
    args.gamma = new_args.gamma
    out_dir = prepare_output_dir(args, args.root_dir)

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(out_dir, 'output.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)
    log.info('===== {} ====='.format(out_dir))


    ''' load regular data '''
    log.info('loading regular training data')
    data = SNLI(args)
    q_vocab = data.TEXT.vocab.itos
    a_vocab = data.LABEL.vocab.itos
    args.char_vocab_size = len(data.char_vocab)
    args.word_vocab_size = len(data.TEXT.vocab)
    args.class_size = len(data.LABEL.vocab)
    args.max_word_len = data.max_word_len

    log.info('loading entropy dev data {}'.format(args.ent_dev))
    with open(args.ent_dev, 'rb') as f:
        ent_dev = pickle.load(f)
    if isinstance(ent_dev[0], list):
        ent_dev = list(itertools.chain(*ent_dev))
    log.info('{} entropy dev examples'.format(len(ent_dev)))
    ent_dev = [[x['data']['premise'], x['data']['hypothesis'], x['data']['label']] for x in ent_dev]

    log.info('loading entropy training data {}'.format(args.ent_train))
    with open(args.ent_train, 'rb') as f:
        ent_train = pickle.load(f)
    if isinstance(ent_train[0], list):
        ent_train = list(itertools.chain(*ent_train))
    log.info('{} entropy training examples'.format(len(ent_train)))
    ent_train = [[x['data']['premise'], x['data']['hypothesis'], x['data']['label']] for x in ent_train]
    train_ent_batches = batchify(ent_train, args.batch_size)
    log.info('{} entropy training batches'.format(len(train_ent_batches)))

    log.info('loading model from {}'.format(args.model_path))
    model = BIMPM(args, data)
    model.load_state_dict(torch.load(args.model_path))
    # model.word_emb.weight.requires_grad = True
    model.cuda(args.gpu)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=args.lr)
    ent_optimizer = optim.Adam(parameters, lr=args.ent_lr)
    criterion = nn.CrossEntropyLoss()

    init_loss, init_acc = evaluate(model, data.dev_iter, args)
    log.info("initial loss {:.4f} accuracy {:.4f}".format(init_loss, init_acc))
    best_acc = init_acc

    dev_ent_batches = batchify(ent_dev, args.batch_size)
    init_ent, init_ent_acc = evaluate_ent(model, dev_ent_batches, args)
    log.info("initial entropy {:.4f} ent_acc {:.4f}".format(init_ent, init_ent_acc))

    epoch = 0
    i_ent, i_mle = 0, 0 # number of examples
    train_loss, train_acc, train_ent = 0, 0, 0
    train_mle_iter = iter(data.train_iter)
    train_ent_iter = iter(train_ent_batches)
    while True:
        model.train()
        for i in range(args.n_ent):
            try:
                prem, hypo, label = next(train_ent_iter)
            except StopIteration:
                random.shuffle(train_ent_batches)
                train_ent_iter = iter(train_ent_batches)
                i_ent = 0
                train_ent = 0
                break
            output = F.softmax(forward(model, prem, hypo, args), 1)
            ent = entropy(output).sum()
            train_ent += ent.data.cpu().numpy()[0]
            loss = - args.gamma * ent
            ent_optimizer.zero_grad()
            loss.backward()
            ent_optimizer.step()
            i_ent += prem.shape[0]
            
        end_of_epoch = False
        for i in range(args.n_mle):
            if i_mle >= len(data.train_iter):
                epoch += 1
                end_of_epoch = True
                data.train_iter.init_epoch()
                train_mle_iter = iter(data.train_iter)
                i_mle = 0
                train_loss = 0
                break
            batch = next(train_mle_iter)
            output = forward(model, batch.premise, batch.hypothesis, args)
            loss = criterion(output, batch.label)
            train_loss += loss.data.cpu().numpy()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i_mle += batch.premise.shape[0]

        if i_mle % 1000 == 0:
            _loss = train_loss / i_mle if i_mle != 0 else 0
            _ent = train_ent / i_ent if i_ent != 0 else 0
            log.info('epoch [{:2}] [{} / {}] loss[{:.5f}] entropy[{:.5f}]'.format(
                epoch, i_mle, len(data.train_iter), _loss, _ent))

        if end_of_epoch or i_mle % 1e5 == 0:
            dev_loss, dev_acc = evaluate(model, data.dev_iter, args)
            dev_ent, dev_ent_acc = evaluate_ent(model, dev_ent_batches, args)
            log.info("dev acc: {:.4f} ent: {:.4f} ent_acc: {:.4f}".format(
                dev_acc, dev_ent, dev_ent_acc))
            model_path = os.path.join(out_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            torch.save(model.state_dict(), model_path)
            if dev_acc > best_acc:
                best_acc = dev_acc
                model_path = os.path.join(out_dir, 'best_model.pt')
                torch.save(model.state_dict(), model_path)
                log.info("best model saved {}".format(dev_acc))

        if epoch > 40:
            break

if __name__ == '__main__':
    main()
