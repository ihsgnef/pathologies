import os
import sys
import random
import logging
import argparse
from shutil import copyfile
from datetime import datetime
import torch

from drqa.model import DocReaderModel
from util import BatchGen, load_data, score


def test():
    from args import conf

    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    args = parser.parse_args()

    # set random seed
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)

    train, dev_x, dev_y, embedding, conf = load_data(conf)

    checkpoint = torch.load(args.baseline)
    # opt = checkpoint['config']
    model = DocReaderModel(vars(conf), embedding, checkpoint['state_dict'])
    model.cuda()

    dev_batches = BatchGen(
            dev_x, batch_size=conf.batch_size,
            pos_size=conf.pos_size, ner_size=conf.ner_size,
            gpu=conf.cuda, evaluation=True)

    predictions = []
    for batch in dev_batches:
        predictions.extend(model.predict(batch))
    em, f1 = score(predictions, dev_y)

    print(em, f1)


def main():
    from args import conf

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=False)
    parser.add_argument('--resume-options', default=False)
    args = parser.parse_args()

    # set random seed
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)

    # setup logger
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler('main.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
            fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    train, dev, dev_y, embedding, conf = load_data(conf)
    log.info(conf)
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(args.resume)
        if args.resume_options:
            conf = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(vars(conf), embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        for i in range(checkpoint['epoch']):
            random.shuffle(list(range(len(train))))  # synchronize random seed
        if conf.reduce_lr:
            for param_group in model.optimizer.param_groups:
                param_group['lr'] *= conf.lr_decay
            log.info('[learning rate reduced by {}]'.format(conf.lr_decay))
    else:
        model = DocReaderModel(vars(conf), embedding)
        epoch_0 = 1

    if conf.cuda:
        model.cuda()

    if args.resume:
        batches = BatchGen(
                dev, batch_size=conf.batch_size,
                pos_size=conf.pos_size, ner_size=conf.ner_size,
                gpu=conf.cuda, evaluation=True)
        predictions = []
        for batch in batches:
            predictions.extend(model.predict(batch))
        em, f1 = score(predictions, dev_y)
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        best_val_score = f1
    else:
        best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + conf.epochs):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(
                train, batch_size=conf.batch_size,
                pos_size=conf.pos_size, ner_size=conf.ner_size,
                gpu=conf.cuda)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % conf.log_per_updates == 0:
                log.info('epoch [{0:2}] updates[{1:6}] \
                        train loss[{2:.5f}] remaining[{3}]'.format(
                    epoch, model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        # eval
        if epoch % conf.eval_per_epoch == 0:
            batches = BatchGen(
                    dev, batch_size=conf.batch_size,
                    pos_size=conf.pos_size, ner_size=conf.ner_size,
                    gpu=conf.cuda, evaluation=True)
            predictions = []
            for batch in batches:
                predictions.extend(model.predict(batch))
            em, f1 = score(predictions, dev_y)
            log.warning("dev EM: {} F1: {}".format(em, f1))
        # save
        if not conf.save_last_only or epoch == epoch_0 + conf.epochs - 1:
            model_file = 'results/baseline_epoch_{}.pt'.format(epoch)
            model.save(model_file, epoch)
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    model_file,
                    os.path.join('results/baseline.pt'))
                log.info('[new best model saved.]')


if __name__ == '__main__':
    test()
