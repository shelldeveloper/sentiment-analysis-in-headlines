"""This script deploys a model to describe the sentiment of news headlines."""
#test
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib

# Check if inputs are present
if len(sys.argv) != 3:
    print("Error: Please provide an input text file and the source of the headline.\n" \
    "Example: python score_headlines.py todaysheadlines.txt nyt")
    sys.exit(1)

# Assign inputs to a variable
input_file      = sys.argv[1]
headline_source = sys.argv[2]

# Load headlines
with open(input_file, 'r', encoding='utf-8') as infile:
    headlines   = infile.readlines()

# Convert headlines to vectors
model           = SentenceTransformer("/opt/huggingface_models/all-MiniLM-L6-v2")
embeddings      = model.encode(headlines)
# embed_file_name = f"headlines_{headline_source}_{file_date}.npy"
# np.save(embed_file_name, embeddings)

# Run vectors through SVM model
# embeddings      = np.load(embed_file_name)
# embeddings      = np.load("headlines_chicagotribune_2024-12-01.npy")
clf             = joblib.load('svm.joblib')
predictions     = clf.predict(embeddings)

# Create output file
txt_file_name   = input_file.rsplit(".", 1)[0]
file_date       = txt_file_name.split("_")[-1]
OUTPUT_FILE     = f"headline_scores_{headline_source}_{file_date}.txt"

with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    for pred, headline in zip(predictions, headlines):
        outfile.write(f"{pred},{headline}\n")
    # for pred in zip(predictions):
    #     outfile.write(f"{pred}\n")

print(f"Predictions saved to: {OUTPUT_FILE}")
