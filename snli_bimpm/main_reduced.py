import os
import copy
import pickle
import random
import numpy as np
import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from bimpm import BIMPM
from dataset import SNLI
from util import prepare_output_dir


def batchify(data, batch_size):
    data = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    batches = []
    for batch in data:
        prem, hypo, label = list(map(list, zip(*batch)))
        prem_len = max(x.shape[0] for x in prem)
        for i, x in enumerate(prem):
            if x.shape[0] < prem_len:
                ones = torch.LongTensor(np.ones((prem_len - x.shape[0],)))
                prem[i] = torch.cat([x, ones], 0)
        hypo_len = max(x.shape[0] for x in hypo)
        for i, x in enumerate(hypo):
            if x.shape[0] < hypo_len:
                ones = torch.LongTensor(np.ones((hypo_len - x.shape[0],)))
                hypo[i] = torch.cat([x, ones], 0)
        prem = torch.stack(prem, 0)
        hypo = torch.stack(hypo, 0)
        label = torch.LongTensor(label)
        batches.append((Variable(prem), Variable(hypo), Variable(label)))
    return batches


def evaluate(model, batches, conf, mode='test'):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in batches:
        prem, hypo, label = batch
        if conf.gpu > -1:
            prem = prem.cuda()
            hypo = hypo.cuda()
            label = label.cuda()
        pred = model(prem, hypo)
        batch_loss = criterion(pred, label)
        loss += batch_loss.data[0]
        _, pred = pred.max(dim=1)
        acc += (pred == label).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().data[0]
    return loss, acc


def train_reduced(model, train_batches, dev_batches, conf):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=conf.lr)
    criterion = nn.CrossEntropyLoss()

    log_dir = os.path.join(conf.out_dir, 'tf_log')
    writer = SummaryWriter(log_dir=log_dir)

    loss = 0
    acc = 0
    size = 0
    max_dev_acc = 0
    batch_i = 0
    epoch_i = 0
    model.train()
    while True:
        batch = train_batches[batch_i]
        prem, hypo, label = batch
        if conf.gpu > -1:
            prem = prem.cuda()
            hypo = hypo.cuda()
            label = label.cuda()

        # limit the lengths of input sentences up to max_sent_len
        if conf.max_sent_len >= 0:
            if prem.size()[1] > conf.max_sent_len:
                prem = prem[:, :conf.max_sent_len]
            if hypo.size()[1] > conf.max_sent_len:
                hypo = hypo[:, :conf.max_sent_len]

        pred = model(prem, hypo)
        optimizer.zero_grad()
        batch_loss = criterion(pred, label)
        loss += batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()

        _, pred = pred.max(dim=1)
        acc += (pred == label).sum().float()
        size += len(pred)

        batch_i += 1

        if batch_i % conf.print_freq == 0:
            train_acc = acc.cpu().data[0] / size
            dev_loss, dev_acc = evaluate(model, dev_batches, conf)
            c = (batch_i + 1) // conf.print_freq
            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('acc/train', train_acc, c)
            writer.add_scalar('acc/dev', dev_acc, c)
            print(f'train loss: {loss:.3f} / \
                    dev loss: {dev_loss:.3f}',
                  f'train acc: {train_acc:.3f} / \
                    dev acc: {dev_acc:.3f}')

            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                best_model = copy.deepcopy(model)

            loss = 0
            model.train()

        if batch_i == len(train_batches):
            random.shuffle(train_batches)
            epoch_i += 1
            batch_i = 0
            loss = 0
            acc = 0
            size = 0
            print('epoch:', epoch_i)
            if epoch_i == conf.epoch:
                ckp_dir = os.path.join(
                        conf.out_dir, 'epoch_{}.pt'.format(epoch_i))
                torch.save(model.state_dict(), ckp_dir)
                break

    writer.close()
    print(f'max dev acc: {max_dev_acc:.3f}')
    return best_model


def main():
    from args import conf

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='results/rawr.train.baseline.pkl')
    parser.add_argument('--dev', default='results/rawr.dev.baseline.pkl')
    parser.add_argument('--truth', default=False)
    args = parser.parse_args()

    conf.train_data = args.train
    conf.dev_data = args.dev

    print('loading regular data...')
    regular_data = SNLI(conf)
    conf.char_vocab_size = len(regular_data.char_vocab)
    conf.word_vocab_size = len(regular_data.TEXT.vocab)
    conf.class_size = len(regular_data.LABEL.vocab)
    conf.max_word_len = regular_data.max_word_len
    conf.out_dir = prepare_output_dir(conf, 'results', 'reduced')

    print('loading reduced data from [{}]'.format(conf.train_data))
    with open(conf.train_data, 'rb') as f:
        train = pickle.load(f)
    print('loading reduced data from [{}]'.format(conf.dev_data))
    with open(conf.dev_data, 'rb') as f:
        dev = pickle.load(f)

    train = [(x['premise'], x['hypothesis'], x['prediction'])
             for ex in train for x in ex['reduced']]
    dev = [(x['premise'], x['hypothesis'], x['label'])
           for ex in dev for x in ex['reduced']]

    train_batches = batchify(train, conf.batch_size)
    dev_batches = batchify(train, conf.batch_size)

    model = BIMPM(conf, regular_data)
    if conf.gpu > -1:
        model.cuda(conf.gpu)

    print('begin training')
    best_model = train_reduced(model, train_batches, dev_batches, conf)

    torch.save(best_model.state_dict(), os.path.join(conf.out_dir, 'best.pt'))
    print('training finished!')


if __name__ == '__main__':
    main()
    # test()
