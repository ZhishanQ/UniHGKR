import pickle
import numpy as np

# Initialize an empty list to store all embedding vectors
all_embeddings = []

model_name_base = 'UniHGKR-7B'
dataset_name = 'compmix_full_set'

# Iterate over all split files
for i in range(8):  # Assume you have 8 split files
    print(i)
    # Construct the path of the split file
    split_path = f"../embedding_cache/test-embeddings-{model_name_base}-size-{dataset_name}_split_{i}_fp16.pkl"

    # Load the embedding vectors from the split file
    with open(split_path, "rb") as fIn:
        embeddings = pickle.load(fIn)

    # Add the embedding vectors to the list
    all_embeddings.append(embeddings)

# Use NumPy's concatenate function to merge all embedding vectors into one large NumPy array
all_embeddings = np.concatenate(all_embeddings, axis=0)

# Save the merged embedding vectors
print('storing')

with open(f"../rag/embedding_cache/test-embeddings-{model_name_base}-size-{dataset_name}_all_fp16.pkl", "wb") as fOut:
    pickle.dump(all_embeddings, fOut)
