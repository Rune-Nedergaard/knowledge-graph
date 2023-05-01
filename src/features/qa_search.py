import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load the distiluse-base-multilingual-cased-v2 model
sentence_model = SentenceTransformer('distiluse-base-multilingual-cased-v2').to(device)

df = pd.read_pickle('data/QA_dataset.pkl')

# Embed all questions in the DataFrame
questions = df['Spørgsmål'].tolist()

# Load or compute question_embeddings
embeddings_path = 'data/qa_embeddings.npy'
if os.path.exists(embeddings_path):
    question_embeddings = np.load(embeddings_path)
else:
    question_embeddings = sentence_model.encode(questions, device=device, show_progress_bar=True)
    np.save(embeddings_path, question_embeddings)

def find_similar_questions(query, embeddings, questions, k=5):
    query_embedding = sentence_model.encode([query], device=device)
    similarities = cosine_similarity(query_embedding, embeddings)
    
    top_k_indices = np.argsort(similarities[0])[::-1][:k]
    top_k_questions = [questions[idx] for idx in top_k_indices]
    top_k_scores = [similarities[0][idx] for idx in top_k_indices]
    
    return top_k_indices, list(zip(top_k_questions, top_k_scores))

# Example usage
new_question ="Hvad er et ofte fremført argument for administrativ centralisering (frem for decentralisering)?" 
top_k = 5

top_k_indices, similar_questions = find_similar_questions(new_question, question_embeddings, questions, k=top_k)
for i, (q, score) in enumerate(similar_questions):
    answer = df.loc[top_k_indices[i], 'Svar']
    print(f"{i + 1}. {q} (Similarity: {score:.4f})\nAnswer: {answer}\n")
