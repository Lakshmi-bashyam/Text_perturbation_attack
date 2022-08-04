import random
import warnings
import json

import torch
from torchtext import data
from torchtext import datasets

SEED = 1234
warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
TEXT = data.Field(tokenize = 'spacy', batch_first=True, include_lengths = True)
LABEL = data.LabelField(dtype = torch.float, batch_first=True)

def data_loaders(batch, device, embedding=True):
    MAX_VOCAB_SIZE = 25000
    BATCH_SIZE = batch
    
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state = random.seed(SEED))
    if embedding:
        TEXT.build_vocab(train_data, 
                        max_size = MAX_VOCAB_SIZE, 
                        vectors = "glove.6B.100d", 
                        unk_init = torch.Tensor.normal_)
    else:
        TEXT.build_vocab(train_data, 
                        max_size = MAX_VOCAB_SIZE)

    LABEL.build_vocab(train_data)
    with open('vocab.json', 'w+') as fp:
        json.dump(TEXT.vocab.stoi, fp)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        device = device,
        sort_within_batch=False,
        shuffle = False)
    
    return (train_iterator, valid_iterator, test_iterator)

def get_vocab():
    return TEXT

# if __name__ == "__main__":
#     a,b,c = data_loaders(64, embedding=False)