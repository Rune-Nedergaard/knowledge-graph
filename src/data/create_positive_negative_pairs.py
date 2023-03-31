import random
import pickle
from tqdm import tqdm

question_chunk_pairs = pickle.load(open("data/question_chunk_pairs.pkl", "rb"))


def create_positive_negative_pairs(question_chunk_pairs, num_negative_samples=1):
    all_paragraphs = [pair[1] for pair in question_chunk_pairs]
    dataset = []
    labels = []

    for question, paragraph in tqdm(question_chunk_pairs, desc="Creating positive and negative pairs"):
        # Positive pair
        dataset.append((question, paragraph))
        labels.append(1)

        # Negative pairs
        for _ in range(num_negative_samples):
            negative_paragraph = random.choice(all_paragraphs)
            while negative_paragraph == paragraph:
                negative_paragraph = random.choice(all_paragraphs)

            dataset.append((question, negative_paragraph))
            labels.append(0)

    return dataset, labels

dataset, labels = create_positive_negative_pairs(question_chunk_pairs, num_negative_samples=1)
pickle.dump(dataset, open("data/fine_tune_dataset.pkl", "wb"))
pickle.dump(labels, open("data/fine_tune_labels.pkl", "wb"))
