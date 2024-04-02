import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import LSTM
import pickle
import lauter
import only_bert

input_size = 300
hidden_size = 768
output_size = len(LSTM.numerical_labels)
max_sequence_length = 50 #Hardcoded because tokenizing might mess with sequence length. Words might be split into multiple tokens

tokenizer = None
with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

glove_embeddings = None
glove_embeddings_matrix = None

with open("glove_embeddings.pickle", "rb") as f:
        glove_embeddings = pickle.load(f)
with open("glove_embeddings_matrix.pickle", "rb") as f:
    glove_embeddings_matrix = pickle.load(f)

test_dataset = LSTM.CustomDataset('../data/validation.masked.txt', glove_embeddings, max_sequence_length)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=LSTM.CustomDataCollator(tokenizer, max_sequence_length), drop_last=True)

# Create an instance of the LSTMModel
#lstm_model = torch.load("lstm_model.pth")

lstm_model = lauter.Lauter(glove_embeddings_matrix, input_size=input_size, hidden_size=hidden_size, output_size=output_size)

model_weights_path = 'lstm_model.pth'
lstm_model.load_state_dict(torch.load(model_weights_path))
lstm_model.to(lstm_model.device)

# Set the model to evaluation mode
lstm_model.eval()

# Lists to store targets and predictions
all_targets = []
all_predictions = []

f = open('../data/validation.masked.categorized.txt')
categories = {} # BATCH SIZE NEEDS TO BE 1 FOR THIS TO WORK

# Iterate over the test dataset
with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(lstm_model.device), targets.to(lstm_model.device)

        category = f.readline().split("\t")[-1].replace("\n", "")
        if category not in categories.keys():
              categories[category] = {"correct": 0, "false": 0}

        # Forward pass
        outputs = lstm_model(inputs)
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


#Calculate accuracy
correct_predictions = sum(1 for target, prediction in zip(all_targets, all_predictions) if (target == prediction).all())
total_samples = len(all_targets)
accuracy = correct_predictions / total_samples
print(f'Accuracy: {accuracy} ({correct_predictions}/{total_samples})')

for category in categories:
      print(f"{category}: {categories[category]['correct']/(categories[category]['correct']+categories[category]['false'])} {categories[category]['correct']}/{(categories[category]['correct']+categories[category]['false'])}")