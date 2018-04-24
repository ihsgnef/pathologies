import torch

from bimpm import BIMPM
from dataset import SNLI
from baseline import evaluate


def load_model(args, data):
    model = BIMPM(args, data)
    model.load_state_dict(torch.load(args.model_path))

    if args.gpu > -1:
        model.cuda(args.gpu)

    return model


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch-size', default=64, type=int)
    # parser.add_argument('--char-dim', default=20, type=int)
    # parser.add_argument('--char-hidden-size', default=50, type=int)
    # parser.add_argument('--dropout', default=0.1, type=float)
    # parser.add_argument('--epoch', default=10, type=int)
    # parser.add_argument('--gpu', default=0, type=int)
    # parser.add_argument('--hidden-size', default=100, type=int)
    # parser.add_argument('--num-perspective', default=20, type=int)
    # parser.add_argument('--use-char-emb', default=False, action='store_true')
    # parser.add_argument('--word-dim', default=300, type=int)
    # parser.add_argument('--model-path', required=True)
    # args = parser.parse_args()
    from args import args
    args.model_path = 'results/baseline.pt'

    data = SNLI(args)

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)

    print('loading model...')
    model = load_model(args, data)

    _, acc = evaluate(model, args, data)

    print(f'test acc: {acc:.3f}')
