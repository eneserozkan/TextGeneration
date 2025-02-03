# TextGeneration
This is just a trial text generation project.

This project creates a Transformer-based text generation model using PyTorch. The model generates new texts using a given initial text. A large text dataset can be used to train the model

# Project Structure
text-generation-transformer/
├── data/
│   ├── text_data.txt                # Training data (large text file)
├── models/
│   ├── transformer_model.py          # Transformer model definition
├── utils/
│   ├── data_preprocessing.py        # Data pre-processing (tokenization, encoding)
│   ├── generate_text.py             # Text generation function
├── main.py                          # Main file that trains the model and generates text
├── config.py                        # Model hyperparameters and training settings
├── requirements.txt                 # Required Python libraries
└── README.md                        

# Installation 
First, install the libraries necessary for the project to work:
pip install -r requirements.txt

# Prepare Training Data
In the file data/text_data.txt add the text data on which the model will be trained. This can be a long book, article or similar large text.

# Train Model and Generate Text
Run main.py to train the model and generate text:
python main.py
This command
	- Divides text into tokens.
	- Trains the transformer model.
	- After completing the training, it generates a new text with a random initial text.

# Working Principle of the Model

1️⃣ Data Preprocessing:
	- Unique characters in the text are identified.
	- Each character is assigned an ID.
	- The text is converted to numeric values and training data is generated.

2️⃣ Transformer Model:
	- Characters are transformed into vectors with an Embedding layer.
	- Multi-head Attention allows the model to learn long-term relationships.
	- Predictions are made with the Linear layer.

3️⃣ Text Generation:
	- The model predicts new characters based on a given initial text.
	- With Softmax & Sampling, the most likely character is selected for each prediction.
	- Text is generated up to the specified length.

# Configuration

The training and model hyperparameters are contained in the config.py file:
config = {
    'embed_size': 256,
    'num_heads': 8,
    'num_layers': 4,
    'hidden_size': 512,
    'seq_length': 30,
    'num_epochs': 500,
    'learning_rate': 0.0001,
}
You can change these parameters according to your needs.
