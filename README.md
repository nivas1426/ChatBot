# ChatBot

This repository contains a chatbot application using Flask and a fine-tuned Sentence Transformer model.

## Project Structure

- `app.py`: Flask application for handling chat requests.
- `data.py`: Functions related to processing corpus.pdf into embeddings.
- `model.py`: Functions related to loading and using the Sentence Transformer model.
- `search.py`: Search functions for querying the corpus.
- `utils.py`: Utility functions for maintaining conversation context.
- `model_train.py`: code related Fine tunning the model.
- `templates/index.html`: The main HTML page for the chatbot.
- `static/`: Directory for static files such as CSS and JavaScript.
- `data/`: Directory containing the corpus and sample question-answer pairs.
- `output/`: Directory containing the fine-tuned Sentence Transformer model.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nivas1426/ChatBot.git
   cd ChatBot
2. Install the required dependencies:
      ```bash
     pip install -r requirements.txt
3. Set up Git LFS for handling large files:
      ```bash
      git lfs install
      git lfs track "output/sentence_transformer_model/model.safetensors"
      git add .gitattributes
      git commit -m "Track large files with Git LFS"
Usage:

#Embeddings of corpus.pdf are generated and model is fine tuned with sample question answers.

1) Run the Flask application:
   ```bash
    python app.py

Open your web browser and go to 'http://127.0.0.1:5002' to interact with the chatbot.


