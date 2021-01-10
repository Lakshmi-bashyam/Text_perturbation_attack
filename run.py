import torch
from torchtext import data
from torchtext import datasets
from dataset import data_loaders, get_vocab
from utils import predict
from model import RNN
import random
import warnings
import string
import collections
import numpy as np
import spacy
from nltk.corpus import stopwords

EMBEDDING_DIM = 100
HIDDEN_DIM = 32
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
N_EPOCHS = 5
BATCH_SIZE = 32
INPUT_DIM = 25002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    stop_words = stopwords.words('english')
    nlp = spacy.load('en')

    # get the data loaders and update vocab
    data_loaders(BATCH_SIZE, device)
    TEXT = get_vocab()

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

    # Check for single text
    text = """I wish I knew what to make of a movie like this. It seems to be divided into two parts -- action sequences and personal dramas ashore. It follows Ashton Kutsher through survival swimmer school, guided by Master Chief Kevin Costner, then to Alaska where a couple of spectacular rescues take place, the last resulting in death.<br /><br />I must say that the scenes on the beach struck me as so stereotypical in so many ways that they should be barnacle encrusted. A typical bar room fight between Navy guys and Coast Guardsmen ("puddle pirates"). The experienced old timer Costner who is, as an elderly bar tender tells him, "married to the Coast Guard." The older chief who "keeps trying to prove to himself that he's still nineteen." The neglected ex wife ashore to whom Kostner pays a farewell visit. The seemingly sadistic demands placed on the swimmers by the instructors, all in pursuit of a loftier goal. The gifted young man hobbled by a troubled past.<br /><br />The problem is that we've seen it all before. If it's Kevin Costner here, it's Clint Eastwood or John Wayne or Lou Gosset Jr. or Vigo Mortenson or Robert DeNiro elsewhere. And the climactic scene has elements drawn shamelessly from "The Perfect Storm" and "Dead Calm." None of it is fresh and none of the old stereotyped characters and situations are handled with any originality.<br /><br />It works best as a kind of documentary of what goes on in the swimmer's school and what could happen afterward and even that's a little weak because we don't get much in the way of instruction. It's mostly personal conflict, romance, and tension about washing out.<br /><br />It's a shame because the U. S. Coast Guard is rather a noble outfit, its official mission being "the safety of lives and property at sea." In war time it is transferred to the Navy Department and serves in combat roles. In World War II, the Coast Guard even managed to have a Medal of Honor winner in its ranks.<br /><br />But, again, we don't learn much about that. We don't really learn much about anything. The film devolves into a succession of visual displays and not too much else. A disappointment."""
    
    output = predict(model, text, TEXT.vocab)

    # Get word importance ranking
    with torch.no_grad():
        ori_op = predict(model, text, TEXT.vocab)
        print("Original output: ", ori_op)
        ranking = {}
        original_text = text
        for word in nlp.tokenizer(text):
            word = word.text
            if word not in string.punctuation and word not in stop_words:
                new_text = original_text.replace(word, '')
                new_op = predict(model, new_text, TEXT.vocab)
                print("Output with"+word+" is "+new_op)
                ranking[word] = np.abs(ori_op - new_op).item()

    print("word importance ranking")
    print(sorted(ranking.items(), key=lambda x: x[1], reverse=True))

    