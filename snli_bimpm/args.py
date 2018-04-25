import argparse

args = argparse.Namespace()

args.dropout = 0.1
args.hidden_size = 100
args.word_dim = 300
args.num_perspective = 20
args.use_char_emb = False
args.char_hidden_size = 50
args.char_dim = 20

args.batch_size = 32
args.gpu = 0
args.epoch = 3
args.lr = 0.001
args.max_sent_len = -1
args.print_freq = 500

rawr_args = argparse.Namespace()


rawr_tune_args = argparse.Namespace()
