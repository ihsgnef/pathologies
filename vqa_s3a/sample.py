import random
import pickle
import data
import config

with open('finals/original_dev.1.pkl', 'rb') as f:
    dev = pickle.load(f)

coco = data.CocoImages(config.val_path)
dev_data = data.get_loader(val=True)
a_vocab = {v:k for k, v in dev_data.dataset.vocab['answer'].items()}
q_vocab = {v:k for k, v in dev_data.dataset.vocab['question'].items()}

correct = []
false = []
for x in dev:
    answers = [a for a, _  in x['labels']]
    if x['original_prediction'] in answers:
        correct.append(x)
    else:
        false.append(x)

random.seed(123)
correct = random.sample(correct, 200)
false = random.sample(false, 200)

correct = {i: x for i, x in enumerate(correct)}
false = {i: x for i, x in enumerate(false)}
for i, x in correct.items():
    coco[x['coco_idx']][1].save('finals/images/correct/{}.png'.format(i))
for i, x in false.items():
    coco[x['coco_idx']][1].save('finals/images/false/{}.png'.format(i))

with open('finals/vqa_human_2class.pkl', 'wb') as f:
    pickle.dump({'correct': correct, 'false': false}, f)
