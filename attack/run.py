import math

import numpy as np
import spacy
import torch
from data.dataset import data_loaders, get_vocab
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

from attack.model import RNN
from attack.utils import (binary_accuracy, get_ranking, predict,
                          replace_with_synonyms)

EMBEDDING_DIM = 100
HIDDEN_DIM = 32
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
N_EPOCHS = 5
BATCH_SIZE = 1


stop_words = stopwords.words('english')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load('en')

# get the data loaders and update vocab
train_iter, val_iter, test_iter = data_loaders(BATCH_SIZE, device, False)
TEXT = get_vocab()
vocab = TEXT.vocab.stoi
INPUT_DIM = len(vocab)
PAD_IDX = vocab['<pad>']

# Intialise data from saved model
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

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

epoch_acc = 0
mod_epoch_acc = 0
for batch in test_iter:
    ip, ip_len = batch.text
    label = batch.label
    ori_op = model(ip, ip_len).squeeze()
    acc = binary_accuracy(ori_op, label)
    epoch_acc += acc.item()

    ranking, ip_text = get_ranking(ip, model, ori_op)
    modified_text = replace_with_synonyms(0.5, model, ranking, sbert_model, ip_text, label)

    mod_op = predict(model, modified_text, vocab)
    mod_acc = binary_accuracy(torch.tensor(mod_op, device=device), label)
    mod_epoch_acc += mod_acc.item()
print("Test accuracy " + epoch_acc / len(test_iter))
print("Test accuracy after attack using glove embedding" + mod_epoch_acc / len(test_iter))

