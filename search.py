from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# Load pre-trained model and corpus embeddings
model_name = 'output/sentence_transformer_model'
model = SentenceTransformer(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

corpus_path = 'data/corpus_texts.txt'
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus_texts = [line.strip() for line in f if line.strip()]

# Load corpus embeddings
def load_corpus_embeddings(corpus_path='data/corpus_embeddings.npy'):
    return torch.tensor(np.load(corpus_path)).to(device)

corpus_embeddings = load_corpus_embeddings()

def search(query):
    # Generate embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)
    
    # Compute cosine similarities between the query and the corpus
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)

    # Find the most similar answers
    top_k = min(5, len(corpus_embeddings))
    top_results = torch.topk(cosine_scores, k=top_k, dim=1)

    # Example: Filter results by score threshold
    score_threshold = 0.5
    results = []
    for idx in top_results.indices[0]:
        score = cosine_scores[0][idx].item()
        if score >= score_threshold:
            results.append({'text': corpus_texts[idx], 'score': score})
    
    return results

if __name__ == "__main__":
    query = "Can you tell me about the unique features of Jessup Cellars' Manny's Blend?"
    results = search(query)
    
    print("Top similar answers (filtered):")
    for result in results:
        print(f"Answer: {result['text']}")
        print(f"Score: {result['score']:.4f}")
        print("-" * 80)
