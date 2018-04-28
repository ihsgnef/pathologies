import argparse

conf = argparse.Namespace()
conf.dropout = 0.1
conf.hidden_size = 100
conf.word_dim = 300
conf.num_perspective = 20
conf.use_char_emb = False
conf.char_hidden_size = 50
conf.char_dim = 20
conf.batch_size = 32
conf.gpu = 0
conf.epoch = 3
conf.lr = 0.001
conf.max_sent_len = -1
conf.print_freq = 500

rawr_conf = argparse.Namespace()
rawr_conf.max_beam_size = 5

tune_conf = argparse.Namespace()
tune_conf.n_mle = 2
tune_conf.n_ent = 2
tune_conf.lr = 0.001
tune_conf.ent_lr = 2e-4
tune_conf.gamma = 1e-3
