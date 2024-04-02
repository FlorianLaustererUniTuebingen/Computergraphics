from transformers import BertTokenizer, BertForMaskedLM, pipeline
from datasets import Dataset
import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils import tensorboard

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

ds = Dataset.load_from_disk("masked_training_dataset")

tokenized_lengths = [len(tokenizer.encode(s)) for s in ds['text']]

# Find the maximum length
max_length = max(tokenized_lengths)
max_length

inputs = tokenizer(ds['text'], return_tensors='pt', max_length=35, truncation=True, padding='max_length')

inputs['labels'] = inputs.input_ids.detach().clone()

labels = tokenizer(ds["label"], add_special_tokens=False)
labels = [item for sublist in labels['input_ids'] for item in sublist]

for tensor in inputs['labels']:
    for n, token in enumerate(tensor):
        if token.item() == tokenizer.mask_token_id:
            tensor[n] = torch.tensor(labels.pop(0))

tokenizer.decode(inputs['input_ids'][0])

inputs['input_ids'][0]
inputs['labels'][0]
tokenizer.decode(3157)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = CustomDataset(inputs)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

#Training

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

model.train()

optim = AdamW(model.parameters(), lr=5e-6)

torch.cuda.is_available()

writer = tensorboard.SummaryWriter()

epochs = 50
step = 0
for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        writer.add_scalar('Loss/train', loss, step)
        
        loss.backward()
        optim.step()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        step += 1

    model.save_pretrained(f'./checkpoint/bert_epoch_{epoch}')
    tokenizer.save_pretrained(f'./checkpoint/bert_epoch_{epoch}')
        
ds_eval = Dataset.load_from_disk("masked_validation_dataset")
eval_list = []
for n, line in enumerate(ds_eval['text']):
    eval_list.append(line.replace("<mask>", "[MASK]"))

def evalModel(path, eval_list, ground_truth):
    fill = pipeline('fill-mask', model=path, tokenizer=path)
    
    results = []
    for line in eval_list:
        results.append(fill(line)[0]['token_str'])

    num_correct = 0
    total_elements = len(results)
    for i in range(total_elements):
        if results[i] == ground_truth[i]:
            num_correct += 1

    proportion_correct = num_correct / total_elements
    return proportion_correct

eval_acc=[]
for n in range(epochs):
    model_path = f"./checkpoint/bert_epoch_{n}"
    eval_acc.append(evalModel(model_path, eval_list, ds_eval['label']))
print(eval_acc)

"""
fill = pipeline('fill-mask', model='./checkpoint/bert_epoch_19', tokenizer='./checkpoint/bert_epoch_10')

results = []
for line in eval_list:
    results.append(fill(line)[0]['token_str'])

num_correct = 0
total_elements = len(results)
for i in range(total_elements):
    if results[i] == ds_eval['label'][i]:
        num_correct += 1
print(num_correct)
print(total_elements)
proportion_correct = num_correct / total_elements
proportion_correct

sequence = f"Positive numbers are more than [MASK] and negative numbers are less than zero."

fill(sequence)[0]['token_str']

ds_eval = Dataset.load_from_disk("masked_validation_dataset")
eval_list = []
for n, line in enumerate(ds_eval['text']):
    eval_list.append(line.replace("<mask>", "[MASK]"))

results = []
for line in eval_list:
    results.append(fill(line)[0]['token_str'])
    
ds_eval['label'][0]

num_correct = 0
total_elements = len(results)

for i in range(total_elements):
    if results[i] == ds_eval['label'][i]:
        num_correct += 1

proportion_correct = num_correct / total_elements
proportion_correct"""