import torch
import os
import pandas
import glob
import io
import torch
from torchtext import data
from torchtext import datasets
from dataset import data_loaders, get_vocab
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import spacy
from model import RNN
import random
import warnings
import string
import collections
import numpy as np
import math

from utils import predict
from utils import get_synonyms
import nltk
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

import json
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
  #print("Getting the perturbed text")
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
        # print(ranking[j][0])
        # print(synlist)
        if len(synlist)-1 < i:
            index = len(synlist)-1
        else:
            index=i
        orig_text = orig_text.replace(ranking[j][0],synlist[index])
    return orig_text

iterations = 700

examples = []
for label in ['pos', 'neg']:
    index = 0
    for fname in glob.iglob(os.path.join('./aclImdb/train', label, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as f:
            print('index: ', index)
            text = f.readline()
            examples.append({'text': text,'label': 0})
            index = index +1
            if (index >= 700):
                break
for label in ['pos', 'neg']:
    index = 0
    for fname in glob.iglob(os.path.join('./aclImdb/test', label, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as f:
            print('index: ', index)
            text = f.readline()
            examples.append({'text': text,'label': 0})
            index = index + 1
            if (index >= 700):
                break

print(len(examples))

perturbed_examples = []
for label in ['pos', 'neg']:
    index = 0
    for fname in glob.iglob(os.path.join('./aclImdb/train', label, '*.txt')):
       # print('here')
        with io.open(fname, 'r', encoding="utf-8") as f:
            print('index: ', index)
            text = f.readline()
            #print('text: ', text)
            perturbed_text = get_perturbed_text(text)
            #print('perturbed text: ', perturbed_text)
            perturbed_examples.append({'text': perturbed_text,'label': 1})
            index = index + 1
            if (index >= 700):
                break

for label in ['pos', 'neg']:
    index = 0
    for fname in glob.iglob(os.path.join('./aclImdb/test', label, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as f:
            print('index: ', index)
            text = f.readline()
            #print('text: ', text)
            perturbed_text = get_perturbed_text(text)
            #print('perturbed text: ', perturbed_text)
            perturbed_examples.append({'text': perturbed_text,'label': 1})
            index = index + 1
            if (index >= 700):
                break

final_data = []
for item in examples:
  final_data.append(item)
for item in perturbed_examples:
  final_data.append(item)

print(len(final_data))

with open('final_data.json', 'w') as fout:
    json.dump(final_data, fout)
print("finished")
