# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable
from .utils import AverageMeter
from .rnn_reader import RnnDocReader
from .distribution import entropy

# Modification:
#   - change the logger name
#   - save & load optimizer state dict
#   - change the dimension of inputs (for POS and NER features)
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

logger = logging.getLogger(__name__)


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()
        self.entropy_loss = AverageMeter()

        # Building network.
        self.network = RnnDocReader(opt, embedding=embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

        ent_lr = opt['ent_lr'] if 'ent_lr' in opt else 2e-3
        self.ent_optimizer = optim.Adamax(parameters, lr=ent_lr,
                                          weight_decay=opt['weight_decay'])

        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            if 'ent_optimizer' in state_dict:
                self.ent_optimizer.load_state_dict(state_dict['ent_optimizer'])




    def update_entropy(self, ex, gamma=0.01):
        self.network.train()
        if self.opt['cuda']:
            inputs = [Variable(e).cuda() for e in ex[:7]]
        else:
            inputs = [Variable(e) for e in ex[:7]]

        self.network.train()
        score_s, score_e = self.network(*inputs, no_softmax=True)
        logits_s = F.softmax(score_s, 1)
        logits_e = F.softmax(score_e, 1)
        ent = entropy(logits_s).sum() + entropy(logits_e).sum()
        loss = - gamma * ent
        self.entropy_loss.update(loss.data[0], ex[0].size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])

        # Update parameters
        self.ent_optimizer.step()
        # self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e).cuda() for e in ex[:7]]
            target_s = Variable(ex[7]).cuda()
            target_e = Variable(ex[8]).cuda()
        else:
            inputs = [Variable(e) for e in ex[:7]]
            target_s = Variable(ex[7])
            target_e = Variable(ex[8])

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.data[0], ex[0].size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, ex, get_all=False):
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e, volatile=True).cuda()
                      for e in ex[:7]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:7]]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        offsets = []
        indices_s, indices_e = [], []
        max_len = self.opt['max_len'] or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            indices_s.append(s_idx)
            indices_e.append(e_idx)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            offsets.append((s_offset, e_offset))
            predictions.append(text[i][s_offset:e_offset])
        if get_all:
            return predictions, offsets, \
                    score_s, score_e, \
                    indices_s, indices_e
        else:
            return predictions

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'ent_optimizer': self.ent_optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
