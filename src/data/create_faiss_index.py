import faiss
import numpy as np
import h5py
import os
import pickle

def load_embeddings(file_path):
    with h5py.File(file_path, 'r') as f:
        # Get the total number of embeddings and their dimension
        num_embeddings = sum([len(group) for group in f.values()])
        dimension = f[next(iter(f))]['0'].shape[0]

        # Initialize the embedding matrix and ID list
        embedding_matrix = np.empty((num_embeddings, dimension), dtype=np.float32)
        ids = []

        # Fill the embedding matrix and ID list
        i = 0
        for filename, group in f.items():
            for index, dataset in group.items():
                embedding_matrix[i] = dataset[:]
                ids.append(f"{filename}_{index}")
                i += 1

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