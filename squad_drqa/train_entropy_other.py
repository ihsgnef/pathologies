import os
import sys
import random
import pickle
import msgpack
import logging
import argparse
import itertools
from datetime import datetime
import scipy
import scipy.stats

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from drqa.model import DocReaderModel
from util import BatchGen, load_data, score
from util import prepare_output_dir

    
def main():
    from args import args
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', required=True)
    # parser.add_argument('--train', required=True)
    # parser.add_argument('--dev', required=True)
    # args.load_model_dir = parser.parse_args().model
    # args.ent_train_dir = parser.parse_args().train
    # args.ent_dev_dir = parser.parse_args().dev
    args.load_model_dir = '/scratch0/shifeng/rawr/drqa/original.pt'
    args.ent_train_dir = 'results/20180217T172242.135276/train.pkl'
    args.ent_dev_dir = 'pkls/original.rawr.dev.pkl'
    args.other_train_dir = 'results/targeted_train_all.pkl'
    out_dir = prepare_output_dir(args, '/scratch0/shifeng/rawr/drqa/')

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

    with open(os.path.join(out_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    log.info('loading regular data from {}'.format(args.data_file))
    train_reg, dev_reg, dev_y, embedding, opt = load_data(args)
    log.info('{} regular training examples'.format(len(train_reg)))
    log.info('{} regular dev examples'.format(len(dev_reg)))
    # log.info(opt)

    ''' load data for regularization '''
    log.info('loading entropy training data from {}'.format(args.ent_train_dir))
    with open(args.ent_train_dir, 'rb') as f:
        train_ent = pickle.load(f)
        if isinstance(train_ent, dict) and 'reduced' in train_ent:
            train_ent = train_ent['reduced']
        if isinstance(train_ent[0][0], list):
            train_ent = list(itertools.chain(*train_ent))

    log.info('loading targeted training data from {}'.format(args.other_train_dir))
    with open(args.other_train_dir, 'rb') as f:
        other_train_ent = pickle.load(f)
        if isinstance(other_train_ent, dict) and 'reduced' in train_ent:
            other_train_ent = other_train_ent['reduced']
        if isinstance(other_train_ent[0][0], list):
            other_train_ent = list(itertools.chain(*other_train_ent))
    train_ent += other_train_ent

    if args.filter_long > 0:
        train_ent = [x for x in train_ent if len(x[5]) < args.filter_long]

    log.info('loading entropy dev data from {}'.format(args.ent_train_dir))
    with open(args.ent_dev_dir, 'rb') as f:
        dev_ent = pickle.load(f)['reduced']
        if isinstance(dev_ent[0], list):
            # dev_ent = list(itertools.chain(*dev_ent))
            dev_ent = [x[0] for x in dev_ent]
        # if args.filter_long > 0:
        #     dev_ent = [x for x in dev_ent if len(x[5]) > args.filter_long]
    log.info('{} entropy training examples'.format(len(train_ent)))
    log.info('{} entropy dev examples'.format(len(dev_ent)))

    log.info('loading model from {}'.format(args.load_model_dir))
    checkpoint = torch.load(args.load_model_dir)
    # opt = checkpoint['config']
    state_dict = checkpoint['state_dict']
    model = DocReaderModel(vars(opt), embedding, state_dict)
    model.cuda()

    ''' initial evaluation '''
    dev_reg_batches = BatchGen(
            dev_reg, batch_size=args.batch_size,
            pos_size=args.pos_size, ner_size=args.ner_size,
            evaluation=True, gpu=args.cuda)
    dev_ent_batches = BatchGen(
            dev_ent, batch_size=args.batch_size,
            pos_size=args.pos_size, ner_size=args.ner_size,
            evaluation=True, gpu=args.cuda)
    predictions = []
    for batch in dev_reg_batches:
        predictions.extend(model.predict(batch))
    em, f1 = score(predictions, dev_y)
    ents, predictions_r = [], []
    for batch in dev_ent_batches:
        p, _, ss, se, _, _ = model.predict(batch, get_all=True)
        ss = ss.cpu().numpy()
        se = se.cpu().numpy()
        ents.append(scipy.stats.entropy(ss.T).sum() + \
                    scipy.stats.entropy(se.T).sum())
        predictions_r.extend(p)
    ent = sum(ents) / len(ents)
    em_r, f1_r = score(predictions_r, dev_y)
    log.info("[dev EM: {:.5f} F1: {:.5f} Ent: {:.5f}]".format(em, f1, ent))
    log.info("[dev EMR: {:.5f} F1R: {:.5f}]".format(em_r, f1_r))
    best_f1_score = f1

    ''' interleaved training '''
    train_ent_batches = BatchGen(
            train_ent, batch_size=args.batch_size,
            pos_size=args.pos_size, ner_size=args.ner_size, gpu=args.cuda)
    len_train_ent_batches = len(train_ent_batches)
    train_ent_batches = iter(train_ent_batches)
    n_reg = 0
    n_ent = 0
    for epoch in range(args.epochs):
        log.warning('Epoch {}'.format(epoch))
        train_reg_batches = BatchGen(
                train_reg, batch_size=args.batch_size,
                pos_size=args.pos_size, ner_size=args.ner_size, gpu=args.cuda)
        start = datetime.now()

        for i_reg, reg_batch in enumerate(train_reg_batches):
            model.update(reg_batch)
            n_reg += 1
            if n_reg > args.start_ent:
                if i_reg % args.n_reg_per_ent == 0:
                    for j in range(args.n_ent_per_reg):
                        try:
                            model.update_entropy(next(train_ent_batches),
                                    gamma=args.gamma)
                            n_ent += 1
                        except StopIteration:
                            n_ent = 0
                            train_ent_batches = iter(BatchGen(
                                train_ent, batch_size=args.batch_size,
                                pos_size=args.pos_size, ner_size=args.ner_size,
                                gpu=args.cuda))

            if n_reg % args.n_report == 0:
                log.info('epoch [{:2}] batch [{}, {}] loss[{:.5f}] entropy[{:.5f}]'.format(
                    epoch, i_reg, n_ent, model.train_loss.avg,
                    -model.entropy_loss.avg / args.gamma))
        
            # if n_reg % args.n_eval == 0:
        dev_reg_batches = BatchGen(
                dev_reg, batch_size=args.batch_size,
                pos_size=args.pos_size, ner_size=args.ner_size,
                evaluation=True, gpu=args.cuda)
        dev_ent_batches = BatchGen(
                dev_ent, batch_size=args.batch_size,
                pos_size=args.pos_size, ner_size=args.ner_size,
                evaluation=True, gpu=args.cuda)

        ''' regular evaluation '''
        predictions = []
        for batch in dev_reg_batches:
            predictions.extend(model.predict(batch))
        em, f1 = score(predictions, dev_y)

        ''' entropy evaluation '''
        ents, predictions_r = [], []
        for batch in dev_ent_batches:
            p, _, ss, se, _, _ = model.predict(batch, get_all=True)
            ss = ss.cpu().numpy()
            se = se.cpu().numpy()
            ents.append(scipy.stats.entropy(ss.T).sum() + \
                        scipy.stats.entropy(se.T).sum())
            predictions_r.extend(p)
        ent = sum(ents) / len(ents)
        em_r, f1_r = score(predictions_r, dev_y)

        log.info("dev EM: {:.5f} F1: {:.5f} Ent: {:.5f}".format(em, f1, ent))
        log.info("[dev EMR: {:.5f} F1R: {:.5f}]".format(em_r, f1_r))

        ''' save best model '''
        if f1 > best_f1_score:
            best_f1_score = f1
            model_file = os.path.join(out_dir, 'best_model.pt')
            model.save(model_file, epoch)
            log.info('[save best model F1: {:.5f}]'.format(best_f1_score))

        ''' save models '''
        model_file = os.path.join(
                out_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
        model.save(model_file, epoch)
        log.info("[save model {}]".format(model_file))

if __name__ == '__main__':
    main()
