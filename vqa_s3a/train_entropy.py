import os
import sys
import pickle
import argparse
import numpy as np
import logging
import scipy
import scipy.stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import data
import model
import config
import utils
from utils import prepare_output_dir
from distribution import entropy

def evaluate(model, data):
    model.eval()
    log_softmax = nn.LogSoftmax().cuda()
    loss, acc, size = 0, 0, 0
    for i, (v, q, a, idx, q_len) in enumerate(data):
        if i > 10:
            break
        v = Variable(v.cuda(async=True))
        q = Variable(q.cuda(async=True))
        a = Variable(a.cuda(async=True))
        q_len = Variable(q_len.cuda(async=True))
        out = model(v, q, q_len)
        nll = -log_softmax(out)
        loss += (nll * a / 10).sum(dim=1).mean().data.cpu()[0]
        acc += utils.batch_accuracy(out.data, a.data).mean()
        size += 1
    return loss / size, acc / size

def evaluate_ent(model, data):
    model.eval()
    ent, acc, size = 0, 0, 0
    for i, (v, q, a, idx, q_len) in enumerate(data):
        if i > 10:
            break
        v = Variable(v.cuda(async=True))
        q = Variable(q.cuda(async=True))
        a = Variable(a.cuda(async=True))
        q_len = Variable(q_len.cuda(async=True))
        out = F.softmax(model(v, q, q_len), 1)
        ent += entropy(out).sum().data.cpu()[0]
        acc += utils.batch_accuracy(out.data, a.data).mean()
        size += 1
    return ent / size, acc / size

def main():
    args = argparse.Namespace()
    args.gamma = 2e-4
    args.n_mle = 2
    args.n_ent = 2
    args.ent_lr = 1e-4

    out_dir = prepare_output_dir(user_dir='/scratch0/shifeng/rawr_data/vqa')
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

    ckp = torch.load('2017-08-04_00.55.19.pth')
    vocab_size = len(ckp['vocab']['question']) + 1
    net = nn.DataParallel(model.Net(vocab_size))
    net.load_state_dict(ckp['weights'])
    net.cuda()
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters)
    ent_optimizer = optim.Adam(parameters, lr=args.ent_lr)
    log_softmax = nn.LogSoftmax().cuda()

    dev_ent_iter = data.get_reduced_loader('results/rawr.dev.pkl', val=True)
    log.info('{} entropy dev examples'.format(len(dev_ent_iter)))
    train_ent_data = data.get_reduced_loader('results/rawr.train.pkl', train=True)
    log.info('{} entropy training examples'.format(len(train_ent_data)))

    dev_mle_iter = data.get_loader(val=True)
    train_mle_data = data.get_loader(train=True)

    dev_loss, dev_acc = evaluate(net, dev_mle_iter)
    log.info('dev loss {:.4f} acc {:.4f}'.format(dev_loss, dev_acc))
    dev_ent, dev_ent_acc = evaluate_ent(net, dev_ent_iter)
    log.info('dev ent {:.4f} acc {:.4f}'.format(dev_ent, dev_ent_acc))
    best_acc = dev_acc
    # best_acc = 0

    epoch = 0
    i_mle, i_ent = 0, 0
    train_loss, train_ent = 0, 0
    size_mle, size_ent = 0, 0
    train_ent_iter = iter(train_ent_data)
    train_mle_iter = iter(train_mle_data)
    while True:
        net.train()
        for i in range(args.n_ent):
            try:
                v, q, a, idx, q_len = next(train_ent_iter)
                v = Variable(v.cuda(async=True))
                q = Variable(q.cuda(async=True))
                a = Variable(a.cuda(async=True))
                q_len = Variable(q_len.cuda(async=True))
            except StopIteration:
                i_ent = 0
                train_ent = 0
                train_ent_iter = iter(train_ent_data)
                break
            out = F.softmax(net(v, q, q_len), 1)
            ent = entropy(out).sum()
            train_ent += ent.data.cpu()[0]
            optimizer.zero_grad()
            ent_optimizer.zero_grad()
            loss = - args.gamma * ent
            loss.backward()
            ent_optimizer.step()
            i_ent += 1
            if i_ent > len(train_ent_data):
                i_ent = 0
                train_ent = 0
                train_ent_iter = iter(train_ent_data)
                break

        end_of_epoch = False
        for i in range(args.n_mle):
            try:
                v, q, a, idx, q_len = next(train_mle_iter)
                v = Variable(v.cuda(async=True))
                q = Variable(q.cuda(async=True))
                a = Variable(a.cuda(async=True))
                q_len = Variable(q_len.cuda(async=True))
            except StopIteration:
                i_mle = 0
                train_loss = 0
                end_of_epoch = True
                epoch += 1
                train_mle_iter = iter(train_mle_data)
                break
            out = F.softmax(net(v, q, q_len), 1)
            nll = -log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            train_loss += loss.data.cpu()[0]
            optimizer.zero_grad()
            ent_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i_mle += 1
            if i_mle > len(train_mle_data):
                i_mle = 0
                train_loss = 0
                end_of_epoch = True
                epoch += 1
                train_mle_iter = iter(train_mle_data)
                break

        if i_mle % 1000 == 0:
            _loss = train_loss / i_mle if i_mle != 0 else 0
            _ent = train_ent / i_ent if i_ent != 0 else 0
            log.info('epoch [{:2}] [{} / {}] loss[{:.5f}] entropy[{:.5f}]'.format(
                epoch, i_mle, len(train_mle_data), _loss, _ent))
    
        if end_of_epoch or i_mle % 1e5 == 0:
            dev_loss, dev_acc = evaluate(net, dev_mle_iter)
            log.info('dev loss {:.4f} acc {:.4f}'.format(dev_loss, dev_acc))
            dev_ent, dev_ent_acc = evaluate_ent(net, dev_ent_iter)
            log.info('dev ent {:.4f} acc {:.4f}'.format(dev_ent, dev_ent_acc))
            model_path = os.path.join(out_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            torch.save(net.state_dict(), model_path)
            if dev_acc > best_acc:
                best_acc = dev_acc
                model_path = os.path.join(out_dir, 'best_model.pt')
                torch.save(net.state_dict(), model_path)
                log.info("best model saved {}".format(dev_acc))

        if epoch > 40:
            break
        
if __name__ == '__main__':
    main()
