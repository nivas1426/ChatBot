from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Load pre-trained model
model_name = 'output/sentence_transformer_model'
model = SentenceTransformer(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def get_embedding(text):
    # Generate embedding for the input text
    embedding = model.encode(text, convert_to_tensor=True).to(device)
    return embedding

def load_corpus_embeddings(corpus_path='data/corpus_embeddings.npy'):
    # Load corpus embeddings from file and move to the same device
    return torch.tensor(np.load(corpus_path)).to(device)


