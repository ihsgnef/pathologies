import copy
import os
import torch

from torch import nn, optim
from time import gmtime, strftime

from bimpm import BIMPM
from dataset import SNLI


def evaluate(model, conf, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        s1, s2 = 'premise', 'hypothesis'
        s1, s2 = getattr(batch, s1), getattr(batch, s2)
        kwargs = {'p': s1, 'h': s2}

        if conf.use_char_emb:
            char_p = torch.LongTensor(data.characterize(s1))
            char_h = torch.LongTensor(data.characterize(s2))

            char_p = char_p.cuda(conf.device)
            char_h = char_h.cuda(conf.device)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        # pred = model(**kwargs)
        pred = model(s1, s2)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data.item()

        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().data.item()
    return loss, acc


def train(conf, data):
    model = BIMPM(conf, data)
    model = model.to(conf.device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=conf.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss, last_epoch = 0, -1
    max_dev_acc, max_test_acc = 0, 0

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == conf.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
            ckp_dir = 'results/baseline_checkpoints/baseline_epoch_{}.pt'
            torch.save(model.state_dict(), ckp_dir.format(present_epoch + 1))
        last_epoch = present_epoch

        s1, s2 = 'premise', 'hypothesis'
        s1, s2 = getattr(batch, s1), getattr(batch, s2)

        # limit the lengths of input sentences up to max_sent_len
        if conf.max_sent_len >= 0:
            if s1.size()[1] > conf.max_sent_len:
                s1 = s1[:, :conf.max_sent_len]
            if s2.size()[1] > conf.max_sent_len:
                s2 = s2[:, :conf.max_sent_len]

        kwargs = {'p': s1, 'h': s2}

        if conf.use_char_emb:
            char_p = torch.LongTensor(data.characterize(s1))
            char_h = torch.LongTensor(data.characterize(s2))

            char_p = char_p.to(conf.device)
            char_h = char_h.to(conf.device)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        # pred = model(**kwargs)
        pred = model(s1, s2)

        optimizer.zero_grad()
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data.item()
        batch_loss.backward()
        optimizer.step()

        if (i + 1) % conf.print_freq == 0:
            dev_loss, dev_acc = evaluate(model, conf, data, mode='dev')
            test_loss, test_acc = evaluate(model, conf, data)
            # c = (i + 1) // conf.print_freq
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
    from args import conf
    print('loading SNLI data...')
    data = SNLI(conf)
    setattr(conf, 'char_vocab_size', len(data.char_vocab))
    setattr(conf, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(conf, 'class_size', len(data.LABEL.vocab))
    setattr(conf, 'max_word_len', data.max_word_len)
    setattr(conf, 'model_time', strftime('%H:%M:%S', gmtime()))

    print('training start!')
    best_model = train(conf, data)

    if not os.path.exists('results'):
        os.makedirs('results')
    torch.save(best_model.state_dict(), 'results/baseline.pt')
    print('training finished!')


def test():
    from args import conf
    data = SNLI(conf)
    setattr(conf, 'char_vocab_size', len(data.char_vocab))
    setattr(conf, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(conf, 'class_size', len(data.LABEL.vocab))
    setattr(conf, 'max_word_len', data.max_word_len)

    model = BIMPM(conf, data)
    model.load_state_dict(torch.load('results/baseline.pt'))
    model = model.to(conf.device)

    _, acc = evaluate(model, conf, data)
    print(f'test acc: {acc:.3f}')


if __name__ == '__main__':
    main()
    test()
