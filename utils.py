import torch
import spacy
nlp = spacy.load('en')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

    
def predict(model, sentence, vocab):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)] 
    # print(tokenized) #tokenize the sentence 
    indexed = [vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction 
    return prediction.item()