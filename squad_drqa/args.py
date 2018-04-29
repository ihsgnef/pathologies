import os
import argparse

conf = argparse.Namespace()
conf.batch_size = 32
conf.concat_rnn_layers = True
conf.cuda = True
conf.doc_layers = 3
conf.dropout_emb = 0.4
conf.dropout_rnn = 0.4
conf.dropout_rnn_output = True
conf.epochs = 40
conf.eval_per_epoch = 1
conf.fix_embeddings = False
conf.grad_clipping = 10
conf.hidden_size = 128
conf.learning_rate = 0.1
conf.log_per_updates = 3
conf.max_len = 15
conf.momentum = 0
conf.ner = True
conf.ner_size = 19
conf.num_features = 4
conf.optimizer = 'adamax'
conf.pos = True
conf.pos_size = 50
conf.question_layers = 3
conf.question_merge = 'self_attn'
conf.reduce_lr = 0.0
conf.rnn_padding = False
conf.rnn_type = 'lstm'
conf.save_last_only = False
conf.seed = 1013
conf.tune_partial = 1000
conf.use_qemb = True
conf.weight_decay = 0
conf.n_report = 100
conf.n_eval = 4000

tune_conf = argparse.Namespace()
tune_conf.start_ent = 400  # number of regular batches before entropy
tune_conf.gamma = 1e-03
tune_conf.learning_rate = 0.1
tune_conf.ent_lr = 2e-4  # default: 2e-3
tune_conf.n_ent_per_reg = 2
tune_conf.n_reg_per_ent = 2
tune_conf.filter_long = 4
