import torch

def encode_text(text, char_to_idx):
    """
    Converts text from character strings to numbers.
    """
    return [char_to_idx[ch] for ch in text]

def prepare_data(text, seq_length):
    """
    Prepares the dataset for training.
    """
    inputs, labels = [], []
    for i in range(len(text) - seq_length):
        inputs.append(text[i:i+seq_length])
        labels.append(text[i+1:i+seq_length+1])
    return torch.tensor(inputs), torch.tensor(labels)