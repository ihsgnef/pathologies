import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import re
import random
import string
from collections import Counter
import torch
import msgpack


def load_data(opt):
    with open('data/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt.pretrained_words = True
    opt.vocab_size = embedding.size(0)
    opt.embedding_dim = embedding.size(1)
    opt.pos_size = len(meta['vocab_tag'])
    opt.ner_size = len(meta['vocab_ent'])
    with open('data/data.msgpack', 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train = sorted(data['train'], key=lambda x: len(x[1]))
    dev = sorted(data['dev'], key=lambda x: len(x[1]))
    dev_y = [x[-1] for x in dev]
    dev = [x[:-1] for x in dev]
    return train, dev, dev_y, embedding, opt


class BatchGen:
    def __init__(self, data, batch_size, pos_size, ner_size, gpu, evaluation=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = batch_size
        self.pos_size = pos_size
        self.ner_size = ner_size
        self.eval = evaluation
        self.gpu = gpu

        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 8
            else:
                assert len(batch) == 10

            context_len = max(len(x) for x in batch[1])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, self.pos_size).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1

            context_ent = torch.Tensor(batch_size, context_len, self.ner_size).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1

            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[6])
            span = list(batch[7])
            if not self.eval:
                y_s = torch.LongTensor(batch[8])
                y_e = torch.LongTensor(batch[9])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1


def is_return_code_zero(args):
    """Return true iff the given command's return code is zero.
    All the messages to stdout or stderr are suppressed.
    """
    with open(os.devnull, 'wb') as FNULL:
        try:
            subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        except subprocess.CalledProcessError:
            # The given command returned an error
            return False
        except OSError:
            # The given command was not found
            return False
        return True


def is_under_git_control():
    """Return true iff the current directory is under git control."""
    return is_return_code_zero(['git', 'rev-parse'])


def prepare_output_dir(args, outdir=None, name=None,
                       argv=None, time_format='%Y%m%dT%H%M%S.%f'):
    """Prepare a directory for outputting training results.

    An output directory, which ends with the current datetime string,
    is created. Then the following infomation is saved into the directory:

        args.txt: command line arguments
        command.txt: command itself
        environ.txt: environmental variables

    Additionally, if the current directory is under git control, the following
    information is saved:

        git-head.txt: result of `git rev-parse HEAD`
        git-status.txt: result of `git status`
        git-log.txt: result of `git log`
        git-diff.txt: result of `git diff`

    Args:
        args (dict or argparse.Namespace): Arguments to save
        outdir (str or None): If str is specified, the output
            directory is created under that path. If not specified, it is
            created as a new temporary directory instead.
        argv (list or None): The list of command line arguments passed to a
            script. If not specified, sys.argv is used instead.
        time_format (str): Format used to represent the current datetime. The
        default format is the basic format of ISO 8601.
    Returns:
        Path of the output directory created by this function (str).
    """
    name_str = datetime.datetime.now().strftime(time_format)
    if name is not None:
        name_str += '_' + name
    if outdir is not None:
        if os.path.exists(outdir):
            if not os.path.isdir(outdir):
                raise RuntimeError(
                    '{} is not a directory'.format(outdir))
        outdir = os.path.join(outdir, name_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=name_str)

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    # Save all the environment variables
    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ)))

    # Save the command
    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    if is_under_git_control():
        # Save `git rev-parse HEAD` (SHA of the current commit)
        with open(os.path.join(outdir, 'git-head.txt'), 'wb') as f:
            f.write(subprocess.check_output('git rev-parse HEAD'.split()))

        # Save `git status`
        with open(os.path.join(outdir, 'git-status.txt'), 'wb') as f:
            f.write(subprocess.check_output('git status'.split()))

        # Save `git log`
        with open(os.path.join(outdir, 'git-log.txt'), 'wb') as f:
            f.write(subprocess.check_output('git log'.split()))

        # Save `git diff`
        with open(os.path.join(outdir, 'git-diff.txt'), 'wb') as f:
            f.write(subprocess.check_output('git diff'.split()))

    return outdir
