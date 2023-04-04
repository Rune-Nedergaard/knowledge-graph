import faiss
import numpy as np
import h5py
import os
import pickle
import hashlib
from tqdm import tqdm

def load_embeddings(file_path):
    with h5py.File(file_path, 'r') as f:
        # Get the total number of embeddings and their dimension
        num_embeddings = sum([len(group) for group in f.values()])
        dimension = f[next(iter(f))]['0'].shape[0]

        # Initialize the embedding matrix and ID list
        embedding_matrix = np.empty((num_embeddings, dimension), dtype=np.float32)
        ids = []

        # Use a set to keep track of unique embeddings
        unique_embeddings = set()

        # Fill the embedding matrix and ID list
        i = 0
        for filename, group in tqdm(f.items(), total = len(f), desc='Loading embeddings'):
            for index, dataset in group.items():
                # Compute a hash of the embedding as a string
                embedding_hash = hashlib.sha256(np.ascontiguousarray(dataset[:]).tobytes()).hexdigest()
                # Add the embedding and ID to the lists if it has not already been seen
                if embedding_hash not in unique_embeddings:
                    unique_embeddings.add(embedding_hash)
                    ids.append(f"{filename}_{index}")
                    embedding_matrix[i] = dataset[:]
                    i += 1

        # Truncate the embedding matrix and ID list to remove duplicates
        embedding_matrix = embedding_matrix[:i]
        ids = ids[:i]

    return embedding_matrix, ids

embedding_matrix, ids = load_embeddings('embeddings.h5')

# Set the parameters for the IVFFlat index
dimension = 768  # Dimension of the BERT embeddings
nlist = 100  # Number of clusters, adjust this value as needed

# Create the IVFFlat index
quantizer = faiss.IndexFlatL2(dimension)
index_ivfflat = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Train the index with the paragraph embeddings
index_ivfflat.train(embedding_matrix)

# Create a mapping dictionary
id_mapping = {original_id: index for index, original_id in enumerate(ids)}

# Convert ids to integers using the mapping
int_ids = [id_mapping[original_id] for original_id in ids]

# Update the index with integer ids
index_ivfflat.add_with_ids(embedding_matrix, np.array(int_ids, dtype=np.int64))

# Save the index to a file
faiss.write_index(index_ivfflat, 'index_ivfflat.faiss')

#Save the mapping dictionary to a file
with open('id_mapping_faiss.pickle', 'wb') as f:
    pickle.dump(id_mapping, f)
