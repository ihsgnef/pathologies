import argparse

args = argparse.Namespace()

args.data_type = 'SNLI'
args.dropout = 0.1
args.hidden_size = 100
args.num_perspective = 20
args.word_dim = 300

args.gpu = 0
args.epoch = 3
args.lr = 0.001
args.max_sent_len = -1
args.print_freq = 500
args.batch_size = 32

args.use_char_emb = False
args.char_hidden_size = 50
args.char_dim = 20
