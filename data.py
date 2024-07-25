import fitz  # PyMuPDF
import json
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

model_name = 'output/sentence_transformer_model'
model = SentenceTransformer(model_name)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_corpus(pdf_path, chunk_size=2000):
    """
    Load text from a PDF file and split into chunks.

    :param pdf_path: Path to the PDF file.
    :param chunk_size: Size of each text chunk.
    :return: A list of text blocks from the PDF.
    """
    if not os.path.isfile(pdf_path):
        logging.error(f"PDF file not found at {pdf_path}")
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    corpus = []
    try:
        document = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text = page.get_text("text")  # Extract text with layout preserved
            if text:
                full_text += text + "\n"  # Append text with new line for separation
        document.close()

        # Clean and chunk the full text
        cleaned_text = clean_text(full_text)
        corpus.extend(chunk_text(cleaned_text, chunk_size))
        logging.info(f"Loaded {len(corpus)} chunks from PDF.")
    except fitz.FitzError as e:
        logging.error(f"Error reading PDF file with PyMuPDF: {e}")
        raise RuntimeError(f"Error reading PDF file: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while reading PDF: {e}")
        raise RuntimeError(f"Unexpected error: {e}")

    return corpus

def clean_text(text):
    """
    Clean up the text from the PDF.

    :param text: Raw text from the PDF.
    :return: Cleaned text.
    """
    # Remove extra whitespace and line breaks
    text = ' '.join(text.split())
    text = text.strip()  # Also remove leading and trailing whitespace
    return text

def chunk_text(text, chunk_size=2000):
    """
    Split the text into chunks of a specified size, ensuring no words are cut off.

    :param text: The text to split.
    :param chunk_size: Size of each chunk.
    :return: List of text chunks.
    """
    chunks = []
    buffer = ""
    
    while len(text) > chunk_size:
        # Find the last space within the chunk size limit
        split_index = text.rfind(' ', 0, chunk_size)
        if split_index == -1:
            split_index = chunk_size
        
        # Include the buffer to ensure complete words
        chunk = buffer + text[:split_index].strip()
        chunks.append(chunk)
        
        # Reset the buffer with the remaining text
        text = text[split_index:].strip()
        buffer = ""
    
    if text:
        chunks.append(buffer + text)
    
    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks

def load_sample_answers(json_path):
    """
    Load sample question answers from a JSON file and convert to a dictionary.

    :param json_path: Path to the JSON file.
    :return: Dictionary with questions as keys and answers as values.
    """
    if not os.path.isfile(json_path):
        logging.error(f"JSON file not found at {json_path}")
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sample_qa = {item.get("question", "").strip(): item.get("answer", "").strip() for item in data if item.get("question") and item.get("answer")}

        logging.info(f"Loaded {len(sample_qa)} Q&A pairs from JSON.")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file: {e}")
        raise RuntimeError(f"Error decoding JSON file: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while reading JSON file: {e}")
        raise RuntimeError(f"Unexpected error: {e}")
    
    return sample_qa

# Example usage
if __name__ == '__main__':
    corpus_path = r'data\Corpus.pdf'  # Change this to your PDF file path
    try:
        corpus_texts = load_corpus(corpus_path)
        logging.info("Corpus Loaded")

        # Save corpus texts to a text file
        with open('data/corpus_texts.txt', 'w', encoding='utf-8') as f:
            for text in corpus_texts:
                f.write(text + "\n")
        logging.info("Corpus texts saved to corpus_texts.txt")

    except Exception as e:
        logging.error(f"Error loading corpus: {e}")

    sample_qa_path = r'data\Sample Question Answers.json'  # Change this to your JSON file path
    try:
        sample_qa = load_sample_answers(sample_qa_path)
        logging.info("Sample Q&A Loaded")
    except Exception as e:
        logging.error(f"Error loading sample Q&A: {e}")
        
    with open('data/corpus_texts.txt', 'r', encoding='utf-8') as f:
        corpus_texts = [line.strip() for line in f if line.strip()]

    # Generate embeddings for the corpus
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

    # Save the embeddings
    np.save('data/corpus_embeddings.npy', corpus_embeddings.cpu().numpy())

