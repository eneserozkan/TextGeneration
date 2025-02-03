import torch
import torch.optim as optim
from models.transformer_model import TextTransformer
from utils.data_preprocessing import encode_text, prepare_data
from utils.generate_text import generate_text
import numpy as np
from config import config

with open('data/complex_long_text.txt', 'r') as file:
    text = file.read()

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encoded_text = encode_text(text, char_to_idx)
inputs, labels = prepare_data(encoded_text, config['seq_length'])

model = TextTransformer(len(chars), config['embed_size'], config['num_heads'], config['num_layers'],
                        config['hidden_size'])
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

for epoch in range(config['num_epochs']):
    model.train()
    optimizer.zero_grad()

    output = model(inputs, labels)
    loss = criterion(output.view(-1, len(chars)), labels.view(-1))

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {loss.item():.4f}")

start_text = "Hello"
generated_text = generate_text(model, start_text, char_to_idx, idx_to_char, length=200, temperature=0.8)
print(generated_text)