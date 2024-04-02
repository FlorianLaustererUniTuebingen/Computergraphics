import pickle
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import gensim
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F

#These are all the possible outputs for the model
numerical_labels = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "no": 11}

def load_glove_embeddings_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        glove_embeddings = pickle.load(file)
    return glove_embeddings

def load_glove_embedding_matrix_from_gensim(glove_model, tokenizer):
    # Extract word embeddings and create the embedding matrix
    embedding_dim = len(glove_model[","])  # Assuming all embeddings have the same dimension
    embedding_matrix = torch.zeros(len(glove_model.index_to_key), embedding_dim)

    for _, token_ids in tokenizer.special_tokens_map.items(): #There might be some words that are represented by two tokens
        embedding_matrix[tokenizer.encode(token_ids, add_special_tokens=False)[0]] = torch.zeros(embedding_dim)

    for word in glove_model.index_to_key:
        if(len(tokenizer.encode(word, add_special_tokens=False))> 5): #There are some very weird words in glove, skip them...
            continue
        for token in tokenizer.encode(word, add_special_tokens=False): #There might be some words that are represented by two tokens
            embedding_matrix[token] = torch.tensor(glove_model[word])
    return embedding_matrix

def load_glove_embedding_from_gensim(gensim_file_path):
    return gensim.models.KeyedVectors.load_word2vec_format(gensim_file_path, binary=True)

class CustomDataset(Dataset):
    def __init__(self, file_path, glove_embeddings, max_sequence_length):
        self.data = []
        self.glove_embeddings = glove_embeddings
        
        self.max_sequence_length = max_sequence_length

        # Load data from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                sentence, target_word = line.strip().split('\t')
                self.data.append((sentence, target_word))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CustomDataCollator:
    def __init__(self, tokenizer, max_sequence_length):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __call__(self, batch):
        sentences, targets = zip(*batch)
        
        # LSTM: Tokenize
        tokens = self.tokenizer(sentences, padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"]
        
        # Map target words to numerical labels
        target_labels = torch.tensor([numerical_labels.get(target_word, len(numerical_labels)) for target_word in targets])
        target_one_hot = one_hot(target_labels, num_classes=len(numerical_labels))

        return input_ids, target_one_hot
    
class FocalLoss(nn.Module):
    """
    Multi-class Focal loss implementation
    """
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, torch.argmax(target, dim=1), self.weight)
        return loss
    
class Lauter(nn.Module):
    def __init__(self, embedding_matrix, input_size, hidden_size, output_size):
        super(Lauter, self).__init__()
        self.dropout_rate = 0.1
        self.bert =  BertForSequenceClassification.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1, dropout=self.dropout_rate)
        self.fc = nn.Linear(1536, 1028) #1536 because BERT has 768 output and we set hidden_size to 768 too
        self.fc_2 = nn.Linear(1028, 256)
        self.fc_4 = nn.Linear(256, output_size)
        self.batchNorm = nn.BatchNorm1d(1536)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        out_bert = self.bert(x)
        out_emb = self.embedding(x)
        out_lstm, _ = self.lstm(out_emb)
        
        pooler = out_bert.hidden_states[-1][:, 0, :] # Get the first token of the last layer, CLS token
        
        out = torch.cat((pooler, out_lstm[:, -1, :]), 1)
        out = self.batchNorm(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.fc_4(out)
    
        return out
