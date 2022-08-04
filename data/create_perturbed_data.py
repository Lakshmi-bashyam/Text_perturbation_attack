import glob
import io
import math
import os
import string
import json

import nltk
import numpy as np
import spacy
import torch
from attack.model import RNN
from attack.utils import get_synonyms, predict
from nltk.corpus import stopwords
from torchtext import data, datasets

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load('en')

TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

MAX_VOCAB_SIZE = 25_000
TEXT.build_vocab(train_data,
                max_size = MAX_VOCAB_SIZE,
                vectors = "glove.6B.100d",
                unk_init = torch.Tensor.normal_)
LABEL.build_vocab(train_data)
vocab = json.load(open('vocab.json'))

EMBEDDING_DIM = 100
HIDDEN_DIM = 32
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
N_EPOCHS = 5
BATCH_SIZE = 1
INPUT_DIM = len(vocab)
PAD_IDX = vocab['<pad>']


model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT
            )
path='tut2-model.pt'
model.load_state_dict(torch.load(path))
model = model.to(device)
model.eval()


def get_perturbed_text(text):

  with torch.no_grad():
    ori_op = predict(model, text, vocab)
    ranking = {}
    original_text = text
    for word in nlp(text):
        if word.text not in string.punctuation and word.text not in stop_words:
            new_text = original_text.replace(word.text, '')
            new_op = predict(model, new_text, vocab)
            ranking[word.text] = {"value": np.abs(ori_op - new_op).item(), "pos": word.pos_}
    ranking = sorted(ranking.items(), key=lambda x: x[1]['value'], reverse=True)

    alpha=0.45
    orig_text = text
    i=1
    for j in range(math.trunc(len(ranking)*alpha)):
        synlist = get_synonyms(ranking[j])
        if len(synlist)-1 < i:
            index = len(synlist)-1
        else:
            index=i
        orig_text = orig_text.replace(ranking[j][0],synlist[index])
    return orig_text

examples = []
for label in ['pos', 'neg']:
   index = 0
   for fname in glob.iglob(os.path.join('./aclImdb/train', label, '*.txt')):
       with io.open(fname, 'r', encoding="utf-8") as f:
           print('index: ', index)
           text = f.readline()
           examples.append({'text': text,'label': 0})
           index = index +1


perturbed_examples = []
for label in ['pos', 'neg']:
    index = 0
    for fname in glob.iglob(os.path.join('./aclImdb/test', label, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as f:
            print('index: ', index)
            text = f.readline()
            print('text: ', text)
            perturbed_text = get_perturbed_text(text)
            print('perturbed text: ', perturbed_text)
            perturbed_examples.append({'text': perturbed_text,'label': 1})
            index = index + 1

final_data = []
for item in examples:
  final_data.append(item)
for item in perturbed_examples:
  final_data.append(item)

with open('final_data_8000.json', 'w') as fout:
    json.dump(final_data, fout)
print("finished")
