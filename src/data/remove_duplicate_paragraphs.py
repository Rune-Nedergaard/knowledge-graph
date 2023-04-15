import os
import torch
import numpy as np
from danlp.models import load_bert_base_model
from scipy.spatial.distance import cosine
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
from tqdm import tqdm
import glob
import hashlib
import h5py
from collections import defaultdict


# Load the Danish BERT model and tokenizer
model = load_bert_base_model()
tokenizer = model.tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



example_paragraphs = [
    "Med venlig hilsen 12 december 2016 Uddannelses- og Forskningsministeriet Børsgade 4 Post Postboks 2135 1015 København K Tel  3392 9700 Fax 3332 3501 Mail ufm@ufm dk Web wwwufmdk CVR-nr Ref-nr 1680 5408 16/056401 Søren Pind Side 1/1",
    "Med venlig hilsen 3 marts 2017 Uddannelses- og Forskningsministeriet Børsgade 4 Post Postboks 2135 1015 København K Tel  3392 9700 Fax 3332 3501 Mail ufm@ufm dk Web wwwufmdk CVR-nr Ref-nr 1680 5408 17/008459-02 Søren Pind Side 1/1",
    "Med venlig hilsen 7 marts 2016 Uddannelses- og Forskningsministeriet Børsgade 4 Post Postboks 2135 1015 København K Tel  3392 9700 Fax 3332 3501 Mail ufm@ufm dk Web wwwufmdk CVR-nr Ref-nr 1680 5408 16/005787-04 Ulla Tørnæs Side 1/1"
]

def calculate_sentence_embeddings(sentences, model):
    with torch.no_grad():
        sentence_embeddings = [model.embed_text(sentence)[1] for sentence in sentences]
    return np.vstack([embedding.numpy().flatten() for embedding in sentence_embeddings])


embeddings = calculate_sentence_embeddings(example_paragraphs, model)

for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        similarity = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))
        print(f"Similarity between paragraph {i + 1} and {j + 1}: {similarity[0][0]:.4f}")

"""
The similarity between these is 0.9834 at least, so I think a threshold of 0.97 would be good.
"""


def read_paragraphs_from_files(file_paths):
    all_paragraphs = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                paragraphs = content.split('\n')
                all_paragraphs.extend(paragraphs)
        except:
            print(f"Could not read file {file_path} as utf-8, trying iso-8859-1")
            try:
                with open(file_path, 'r', encoding='iso-8859-1') as file:
                    content = file.read()
                    paragraphs = content.split('\n')
                    all_paragraphs.extend(paragraphs)
            except:
                print(f"Could not read file {file_path} as iso-8859-1 either, skipping")
                continue
            
    return all_paragraphs



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


def remove_duplicates(paragraphs, ids, embeddings_matrix, threshold):
    """
    This function intends to remove headers, footnotes and other duplicate paragraphs from the dataset.
    Since there are cases where a paragraph is similar and supposed to be so (such as questions being reiterated),
    we only remove paragraphs that are very similar at least 5 other paragraphs.
    """
    embedding_size = embeddings_matrix.shape[1]
    unique_paragraphs_index = AnnoyIndex(embedding_size, "angular")

    # Add all embeddings to the index
    for idx, embedding in enumerate(embeddings_matrix):
        unique_paragraphs_index.add_item(idx, embedding)

    unique_paragraphs_index.build(10)

    unique_paragraphs = set()
    unique_paragraph_ids = set()

    for idx, (paragraph, embedding) in enumerate(zip(paragraphs, embeddings_matrix)):
        nearest_indices, distances = unique_paragraphs_index.get_nns_by_vector(embedding, n=6, search_k=-1, include_distances=True)

        # Check if there are 5 or more similar paragraphs
        similar_count = sum(dist < threshold for dist in distances[1:])

        if similar_count < 5:
            unique_paragraphs.add(paragraph)
            unique_paragraph_ids.add(ids[idx])

    return list(unique_paragraphs), list(unique_paragraph_ids)






# Set the input and output directories
input_folder = 'data/all_paragraphs/paragraphs'
output_folder = 'data/all_paragraphs_filtered'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of all txt files in the input folder
input_files = glob.glob(os.path.join(input_folder, '*.txt'))

# Load the precomputed embeddings
embeddings_matrix, ids = load_embeddings('embeddings.h5')

# Read paragraphs from all input files
all_paragraphs = read_paragraphs_from_files(input_files)

# Remove duplicate paragraphs
euclidean_threshold = 2 * (1 - 0.98)
unique_paragraphs, unique_paragraph_ids = remove_duplicates(all_paragraphs, ids, embeddings_matrix, threshold=euclidean_threshold)

## Save the unique paragraphs
output_folder = 'data/subset_paragraphs_filtered'

# Organize unique paragraphs by their base filenames
unique_paragraphs_by_basename = defaultdict(list)

# Separate base filenames and indices from unique_paragraph_ids
for unique_id in unique_paragraph_ids:
    basename, index = unique_id.split('_')
    index = int(index)
    unique_paragraphs_by_basename[basename].append(index)

print(f"Found {len(unique_paragraphs_by_basename)} unique paragraphs")

# Write the filtered paragraphs back to the output folder
for basename, indices in unique_paragraphs_by_basename.items():
    try:
        input_file_path = os.path.join(input_folder, basename)
        output_file_path = os.path.join(output_folder, basename)

        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            paragraphs = input_file.readlines()

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for idx, paragraph in enumerate(paragraphs):
                if idx in indices:
                    output_file.write(paragraph)
    except:
        print(f"Could not read file {input_file_path} as utf-8, trying iso-8859-1")
        try:
            with open(input_file_path, 'r', encoding='iso-8859-1') as input_file:
                paragraphs = input_file.readlines()

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for idx, paragraph in enumerate(paragraphs):
                    if idx in indices:
                        output_file.write(paragraph)
        except:
            print(f"Could not read file {input_file_path} as iso-8859-1 either, skipping")
            continue

print(f"Filtered paragraphs saved in {output_folder}")
