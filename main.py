import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Params

model_path = "./models/best_model.pth"
csv_file_path = 'articles.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 1024
nhead = 8
num_layers = 6
num_epochs = 200
learning_rate = 0.001
max_len = 1024

# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.vocab_size = len(tokenizer)
        self.tokenized_texts = [self.encode(text) for text in texts]
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        padded_tokens = tokens + [self.tokenizer['<PAD>']] * (self.seq_len - len(tokens))
        return torch.tensor(padded_tokens[:self.seq_len], dtype=torch.long)
    
    def encode(self, text):
        return [self.tokenizer[word] for word in text.split()]

# Load data / create tokenizer
df = pd.read_csv(csv_file_path, encoding='utf-8')
texts = df['text'].tolist()

tokenizer = {word: idx for idx, word in enumerate(set(' '.join(texts).split()))}
tokenizer['<PAD>'] = len(tokenizer)
tokenizer['<SOS>'] = len(tokenizer) + 1
tokenizer['<EOS>'] = len(tokenizer) + 2

# Create dataset / dataloader
dataset = TextDataset(texts, tokenizer, seq_len=10)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model class
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(MiniGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))  # Maximum length adjustable
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=0.1)
        encoder_layers.self_attn.batch_first = True
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :]
        x = self.embedding(x) + pos_enc
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x

# Init model, optimizer, criterion
vocab_size = len(tokenizer)

model = MiniGPT(vocab_size, d_model, nhead, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'Loss': loss.item()})
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), model_path)
            print("Model updated!")

def generate_text(model, tokenizer, start_text, max_len=128, temperature=1):
    model.eval()
    start_tokens = [tokenizer[word] for word in start_text.split()]
    generated_text = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            outputs = model(generated_text)
            logits = outputs[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == tokenizer['<EOS>']:
                break
            generated_text = torch.cat((generated_text, next_token), dim=1)

    decoded_text = ' '.join([list(tokenizer.keys())[token] for token in generated_text.squeeze().tolist()])
    return decoded_text

# Train
train(model, dataloader, criterion, optimizer, epochs=num_epochs)

# Generate text
generated_text = generate_text(model, tokenizer, start_text="The machine learning course")
print(generated_text)
