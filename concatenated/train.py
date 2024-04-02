import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import lauter
from transformers import BertTokenizer
import torch.nn.functional as F
import pickle

CLEAN_START = True #Set this to false if you already have a tokenizer and an embedding matrix, 

batch_size = 2
epochs = 50
input_size = 300
hidden_size = 768
output_size = len(lauter.numerical_labels)
learning_rate = 0.01
max_sequence_length = 50 #Hardcoded because tokenizing might mess with sequence length. Words might be split into multiple tokens

tokenizer = None
if CLEAN_START:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=max_sequence_length)
    with open("tokenizer.pickle", "wb") as f: #Save tokenizer so it can be reused in the next training
        pickle.dump(tokenizer, f)
else:
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
print("created tokenizer")

glove_embeddings = None
glove_embeddings_matrix = None

if CLEAN_START:
    glove_embeddings = lauter.load_glove_embedding_from_gensim('v_glove_300d_2.0')
    glove_embeddings_matrix = lauter.load_glove_embedding_matrix_from_gensim(glove_embeddings, tokenizer)
    with open("glove_embeddings.pickle", "wb") as f: #Save embeddings and embedding matrix so it can be reused in the next training
        pickle.dump(glove_embeddings, f)
    with open("glove_embeddings_matrix.pickle", "wb") as f:
        pickle.dump(glove_embeddings_matrix, f)
else:
    with open("glove_embeddings.pickle", "rb") as f:
        glove_embeddings = pickle.load(f)
    with open("glove_embeddings_matrix.pickle", "rb") as f:
        glove_embeddings_matrix = pickle.load(f)
print("created embedding matrix")

#training dataset
dataset = lauter.CustomDataset("../data/train.masked.txt", glove_embeddings, max_sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lauter.CustomDataCollator(tokenizer, max_sequence_length), drop_last=True)

#validation dataset
test_dataset = lauter.CustomDataset('../data/validation.masked.txt', glove_embeddings, max_sequence_length)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lauter.CustomDataCollator(tokenizer, max_sequence_length), drop_last=True)

print("created dataloader")

model = lauter.Lauter(glove_embeddings_matrix, input_size, hidden_size, output_size)

#criterion = lauter.FocalLoss()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

scheduler = StepLR(optimizer, step_size=10, gamma=0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#Freeze embeddings for the first few epochs
model.embedding.weight.requires_grad = False 

print("start training...")
for epoch in range(epochs):
    model.train()
    if(epoch == 5):
        #unfreeze the embedding layer
        model.embedding.weight.requires_grad = True
        #We need to reinitialize the optimizer after unfreezing the embeddings
        optimizer = Adam(model.parameters(), lr = learning_rate)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    for inputs, targets in dataloader: #training loop
        optimizer.zero_grad()
        targets, inputs = targets.to(device), inputs.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets.float())

        loss.backward()
        optimizer.step()

    scheduler.step()

    model.eval() #Switch model to eval so we can check the accuracy for every epoch
    categories = {} # BATCH SIZE IN VALIDATION DATALOADER NEEDS TO BE 1 FOR THIS TO WORK
    all_targets = []
    all_predictions = []
    validation_file = open('../data/validation.masked.categorized.txt')
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            category = validation_file.readline().split("\t")[-1].replace("\n", "")
            if category not in categories.keys():
                  categories[category] = {"correct": 0, "false": 0}

            # Forward pass
            outputs = model(inputs)
            # Convert outputs to predictions
            predictions = F.softmax(outputs, dim=1).argmax(dim=1)
            targets = targets.argmax(dim=1)

            if(targets.cpu().numpy() == predictions.cpu().numpy()).all():
                  categories[category]["correct"] += 1
            else:
                  categories[category]["false"] += 1

            # Append targets and predictions to lists
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    validation_file.close()


    #Calculate accuracy
    correct_predictions = sum(1 for target, prediction in zip(all_targets, all_predictions) if (target == prediction).all())
    total_samples = len(all_targets)
    accuracy = correct_predictions / total_samples
    print(f'Accuracy: {accuracy} ({correct_predictions}/{total_samples})')

    for category in categories:
          print(f"{category}: {categories[category]['correct']/(categories[category]['correct']+categories[category]['false'])} {categories[category]['correct']}/{(categories[category]['correct']+categories[category]['false'])}")

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Learning rate: {scheduler.get_last_lr()[0]}')

#torch.save(lstm, "lstm_model.pth")
torch.save(model.state_dict(), "lstm_model.pth") #save model for further use
