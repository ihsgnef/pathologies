import os
import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from nltk import word_tokenize


class SNLI():

    def __init__(self, conf):
        self.TEXT = data.Field(batch_first=True, tokenize=word_tokenize,
                               lower=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        print('loading data from {}'.format('data/'))
        self.train, self.dev, self.test = datasets.SNLI.splits(
                self.TEXT, self.LABEL, root='data/')

        vector_cache = '.vector_cache/vocab.vectors.pt'
        if os.path.isfile(vector_cache):
            print('loading vector cache from {}'.format(vector_cache))
            self.TEXT.build_vocab(self.train, self.dev, self.test)
            self.TEXT.vocab.vectors = torch.load(vector_cache)
        else:
            self.TEXT.build_vocab(self.train, self.dev, self.test,
                                  vectors=GloVe(name='840B', dim=300,
                                                cache='../glove'))
            torch.save(self.TEXT.vocab.vectors, vector_cache)
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[conf.batch_size] * 3,
                                       device=conf.device)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])
        # for <pad>
        self.char_vocab = {'': 0}
        # for <unk> and <pad>
        self.characterized_words = [[0 for _ in range(self.max_word_len)],
                                    [0 for _ in range(self.max_word_len)]]

        if conf.use_char_emb:
            self.build_char_vocab()

    def build_char_vocab(self):
        # for normal words
        for word in self.TEXT.vocab.itos[2:]:
            chars = []
            for c in list(word):
                if c not in self.char_vocab:
                    self.char_vocab[c] = len(self.char_vocab)

                chars.append(self.char_vocab[c])

            chars.extend([0] * (self.max_word_len - len(word)))
            self.characterized_words.append(chars)

    def characterize(self, batch):
        """
        :param batch: Pytorch Variable with shape (batch, seq_len)
        :return: Pytorch Variable with shape (batch, seq_len, max_word_len)
        """
        batch = batch.data.cpu().numpy().astype(int).tolist()
        return [[self.characterized_words[w] for w in words]
                for words in batch]
