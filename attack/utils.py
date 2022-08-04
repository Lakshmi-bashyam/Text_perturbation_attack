import json
import math
import string
from collections import OrderedDict

import nltk
import numpy as np
import spacy
import torch
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

stop_words = stopwords.words('english')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from nltk.corpus import wordnet as wn

nlp = spacy.load('en')
vocab = json.load(open('vocab.json'))
reverse_vocab = {}
for k,v in vocab.items():
    reverse_vocab[v] = k


glove_preTrained=api.load("glove-wiki-gigaword-50")

def get_gloveWord(word):
    if word[0] in glove_preTrained.wv.vocab:
        syn = glove_preTrained.most_similar(word[0])
        syn = [item[0] for item in syn]
    else:
        syn = [word[0]]
    return syn 

def binary_accuracy(preds, y):
    
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

    
def predict(model, sentence, vocab):
    tokenized = [tok.text for tok in nlp(sentence)] 
    indexed = [vocab.get(t, 0) for t in tokenized]             #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()

def wordnet_similarity(word):
    """[Retruns list of synonyms for a word extracted from wordnet corpus]

    Args:
        word ([str]): [word to replace]

    Returns:
        [list]: [synonyms list]
    """ 
    print(word[0])
    if word[1]['pos'] == 'VERB':
        pos = wn.VERB
    elif word[1]['pos'] == 'ADJ':
        pos = wn.ADJ
    elif word[1]['pos'] == 'ADV':
        pos = wn.ADV
    elif word[1]['pos'] == 'NOUN':
        pos = wn.NOUN
    else:
        return [word[0]]
    
    synonyms = []
    for syn in wn.synsets(word[0], pos=pos): 
        for l in syn.lemmas():
            # print(l)
            synonyms.append(l.name().replace("_", " "))
    return set(synonyms)


def get_ranking(ip, model, ori_op):
    ranking = OrderedDict()
    ip_text = []
    for index, word in enumerate(ip.cpu()[0]):
        replace_index = torch.ones(len(ip[0]), dtype=torch.bool)
        word = reverse_vocab[word.item()]
        ip_text.append(word)
        if word not in string.punctuation and word not in stop_words and word != '<br />':
            replace_index[index] = False
            new_ip = ip[0][replace_index]
            new_ip = new_ip.view(1,len(new_ip))
            new_op = model(new_ip, torch.tensor([len(new_ip[0])], device=device)).squeeze()
            ranking[word] = torch.abs(ori_op - new_op).item()
    pos = {}
    for text in nlp(" ".join(ip_text)):
        pos[text.text] = str(text.pos_)
    ranking_with_pos= {}
    for word, rank in ranking.items():
        ranking_with_pos[word] = {"value": rank, "pos": pos.get(word, "")}
    return(sorted(ranking_with_pos.items(), key=lambda x: x[1]['value'], reverse=True), " ".join(ip_text))

def replace_with_synonyms(alpha, model, ranking, embed_model, orig_text, orig_label):

    synlist = {}
    mod_text = orig_text
    key_list = [ranking[i][0] for i in  range(math.trunc(len(ranking)*alpha))]
    syn_list = [get_gloveWord(ranking[i]) for i in  range(math.trunc(len(ranking)*alpha))]

    for i in range(max([len(x) for x in syn_list])):
        mod_text = orig_text
        for index, word in enumerate(key_list):
            if len(syn_list[index])-1 < i:
                mod_text = mod_text.replace(word, syn_list[index][-1])
            else:
                mod_text = mod_text.replace(word, syn_list[index][i])

        if embedding_constraint(orig_text, mod_text, embed_model, 0.65):
            pred = predict(model, mod_text, vocab)
            if not bool(round(pred)) == bool(orig_label.cpu()[0]):
                return mod_text
    return orig_text
  
def get_synonyms(word):
    synonyms = []
    if word[1]['pos'] == 'VERB':
        pos = wn.VERB
    elif word[1]['pos'] == 'ADJ':
        pos = wn.ADJ
    elif word[1]['pos'] == 'ADV':
        pos = wn.ADV
    elif word[1]['pos'] == 'NOUN':
        pos = wn.NOUN
    else:
        return [word[0]]
    for syn in wn.synsets(word[0], pos=pos): 
        for l in syn.lemmas():
            synonyms.append(l.name().replace("_", " "))
    if not synonyms:
        synonyms.append(word[0])
    return list(set(synonyms)) 

def embedding_constraint(original_txt, mod_txt, embed_model, threshold):

    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    for sent1, sent2 in zip(nltk.sent_tokenize(original_txt), nltk.sent_tokenize(mod_txt)):
        t1 = embed_model.encode(sent1)
        t2 = embed_model.encode(sent2)
        if(cosine(t1, t2) < threshold):
            return False
    return True
