import torch


def generate_text(model, start_str, char_to_idx, idx_to_char, length=100, temperature=1.0):
    """
    Text generation function using the Transformer model.
    """
    model.eval()
    input_str = [char_to_idx[ch] for ch in start_str]
    input_tensor = torch.tensor(input_str).unsqueeze(0)

    generated_text = start_str

    for _ in range(length):
        output = model(input_tensor, input_tensor)
        output = output.squeeze(0)

        output = output / temperature
        probabilities = torch.nn.functional.softmax(output[-1], dim=0)
        predicted_idx = torch.multinomial(probabilities, 1).item()

        generated_text += idx_to_char[predicted_idx]
        input_tensor = torch.tensor([predicted_idx]).unsqueeze(0)

    return generated_text