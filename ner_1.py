import nltk
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

EPOCH = 20
dataset = json.loads(open('data/ner/merged_5k.json', 'r').read())

# Download the Punkt tokenizer models
nltk.download('punkt')

def preprocess_data(dataset):
    preprocessed_data = []
    
    for entry in dataset:
        sample = entry['sample']
        output = entry['output']
        
        # Tokenize the sample
        tokens = sample.split(' ')
        
        # Initialize an empty label list with 'O' (denoting no entity)
        labels = ['O'] * len(tokens)
        
        # Create a mapping of entity names to their tags and entity categories
        entity_map = {}

        for entity in output:
            if 'name' in entity and 'tag' in entity:
                entity_map[entity['name']] = (entity['tag'], entity['entity'])

        # Assign labels to tokens based on the entity map
        for i, token in enumerate(tokens):
            for entity_name, (tag, entity_category) in entity_map.items():
                if entity_name.startswith(token):
                    labels[i] = f'B-{tag}'
                    break
                elif entity_name.find(token) > 0:
                    labels[i] = f'I-{tag}'
                    break
        
        preprocessed_data.append((tokens, labels))
    
    return preprocessed_data

# Preprocess the data
preprocessed_data = preprocess_data(dataset)


all_tokens = [token for sample, labels in preprocessed_data for token in sample]
all_labels = [label for sample, labels in preprocessed_data for label in labels]

vocab = list(set(all_tokens))
label_encoder = LabelEncoder()
label_encoder.fit(list(set(all_labels)))

# Convert tokens and labels to numerical values
numerical_data = [(list(map(vocab.index, sample)), list(label_encoder.transform(labels))) for sample, labels in preprocessed_data]

# Split the data into training, validation, and test sets
train_data, test_data = train_test_split(numerical_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # Separate inputs and labels
    inputs = [torch.tensor(item[0]) for item in batch]
    labels = [torch.tensor(item[1]) for item in batch]
    
    # Pad sequences
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return inputs, labels

# Create data loaders
train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)
test_dataset = CustomDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)  # Removed the view method, as it's not necessary with batch_first=True
        tag_space = self.fc(lstm_out.contiguous().view(-1, lstm_out.shape[2]))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = SimpleLSTM(vocab_size=len(vocab), embed_size=100, hidden_size=100, output_size=len(label_encoder.classes_))

loss_function = nn.NLLLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Training loop

for epoch in range(EPOCH):  # Number of epochs
    for sentences, labels in train_loader:
        sentences = sentences.long()
        labels = labels.view(-1).long()  # Reshape the labels to match the shape of the data

        # Forward pass
        model.zero_grad()
        tag_scores = model(sentences)
        
        # Compute the loss, gradients, and update the parameters
        loss = loss_function(tag_scores, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    with torch.no_grad():
        total_loss = 0
        for sentences, labels in val_loader:
            sentences = sentences.long()
            labels = labels.view(-1).long()  # Reshape the labels to match the shape of the data

            tag_scores = model(sentences)
            loss = loss_function(tag_scores, labels)
            total_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {total_loss / len(val_loader)}')


# Switch the model to evaluation mode
model.eval()

# Initialize variables to store the total loss and the number of correct predictions
total_loss = 0
total_correct = 0
total_labels = 0

# Initialize a confusion matrix if you plan to calculate metrics like F1-score later

# No need to compute gradients during testing
with torch.no_grad():
    for sentences, labels in test_loader:
        sentences = sentences.long()
        labels = labels.view(-1).long()
        
        # Forward pass
        tag_scores = model(sentences)
        
        # Compute the loss
        loss = loss_function(tag_scores, labels)
        total_loss += loss.item()
        
        # Get the predictions
        predictions = torch.argmax(tag_scores, dim=1)
        
        # Calculate the number of correct predictions
        total_correct += (predictions == labels).sum().item()
        total_labels += labels.size(0)
        

# Calculate the average loss and accuracy over all batches
average_loss = total_loss / len(test_loader)
accuracy = total_correct / total_labels

print(f'Test Loss: {average_loss}')
print(f'Test Accuracy: {accuracy * 100:.2f}%')

