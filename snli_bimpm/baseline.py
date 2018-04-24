import argparse
import copy
import os
import torch

from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from bimpm import BIMPM
from dataset import SNLI


def evaluate(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1:
                char_p = char_p.cuda(args.gpu)
                char_h = char_h.cuda(args.gpu)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        # pred = model(**kwargs)
        pred = model(s1, s2)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]

        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().data[0]
    return loss, acc


def train(args, data):
    model = BIMPM(args, data)
    if args.gpu > -1:
        model.cuda(args.gpu)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_acc, max_test_acc = 0, 0

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch

        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s2 = getattr(batch, s1), getattr(batch, s2)

        # limit the lengths of input sentences up to max_sent_len
        if args.max_sent_len >= 0:
            if s1.size()[1] > args.max_sent_len:
                s1 = s1[:, :args.max_sent_len]
            if s2.size()[1] > args.max_sent_len:
                s2 = s2[:, :args.max_sent_len]

        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1:
                char_p = char_p.cuda(args.gpu)
                char_h = char_h.cuda(args.gpu)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        # pred = model(**kwargs)
        pred = model(s1, s2)

        optimizer.zero_grad()
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            dev_loss, dev_acc = evaluate(model, args, data, mode='dev')
            test_loss, test_acc = evaluate(model, args, data)
            c = (i + 1) // args.print_freq

            # writer.add_scalar('loss/train', loss, c)
            # writer.add_scalar('loss/dev', dev_loss, c)
            # writer.add_scalar('acc/dev', dev_acc, c)
            # writer.add_scalar('loss/test', test_loss, c)
            # writer.add_scalar('acc/test', test_acc, c)

            print(f'train loss: {loss:.3f} / \
                    dev loss: {dev_loss:.3f} / \
                    test loss: {test_loss:.3f} /'
                  f'dev acc: {dev_acc:.3f} / \
                    test acc: {test_acc:.3f}')

            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)

            loss = 0
            model.train()

    # writer.close()
    print(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')

    return best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--data-type', default='SNLI')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--max-sent-len', default=-1, type=int)
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--print-freq', default=500, type=int)
    parser.add_argument('--use-char-emb', default=False, action='store_true')
    parser.add_argument('--word-dim', default=300, type=int)
    args = parser.parse_args()

    if args.data_type == 'SNLI':
        print('loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('loading Quora data...')
        data = Quora(args)
    else:
        raise NotImplementedError('only SNLI or Quora data is possible')

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))

    print('training start!')
    best_model = train(args, data)

    if not os.path.exists('results'):
        os.makedirs('results')
    torch.save(best_model.state_dict(),
               f'results/baseline.pt')

    print('training finished!')


if __name__ == '__main__':
    main()
