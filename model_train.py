import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentencesDataset, LoggingHandler
from torch.utils.data import DataLoader
import logging
import os

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Load pre-trained Sentence-BERT model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Load question-answer pairs from the JSON file
sample_qa_path = r'data/Sample Question Answers.json'  # Change this to your JSON file path
if not os.path.isfile(sample_qa_path):
    logging.error(f"JSON file not found at {sample_qa_path}")
    raise FileNotFoundError(f"JSON file not found at {sample_qa_path}")

with open(sample_qa_path, 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)

# Convert the question-answer pairs into InputExample format
train_examples = []
for qa in qa_pairs:
    train_examples.append(InputExample(texts=[qa['question'], qa['answer']], label=1.0))

# Create a DataLoader
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

# Define the loss function
train_loss = losses.CosineSimilarityLoss(model)

# Ensure output directory exists
output_dir = 'output/sentence_transformer_model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=15,
          warmup_steps=100,
          output_path=output_dir)

# Save the model
model.save(output_dir)
logging.info(f"Model saved to {output_dir}")
