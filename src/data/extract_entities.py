import os
import nltk
from danlp.models import load_flair_ner_model
from flair.data import Sentence
from tqdm import tqdm

# Download the NLTK Punkt tokenizer if not already downloaded
nltk.download('punkt')

# Define directories
input_dir = 'data/processed/'
output_dir = 'data/entities/'

# Load the Flair NER model
flair_model = load_flair_ner_model()

# Process files
for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
    if not filename.endswith('.txt'):
        continue

    with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
        text = f.read()

    # Split text into sentences using NLTK
    sentences = nltk.sent_tokenize(text)

    # Extract entities for each sentence
    all_entities = []
    for sentence in sentences:
        flair_sentence = Sentence(sentence)
        flair_model.predict(flair_sentence)
        #print(flair_sentence.to_tagged_string())

        for span in flair_sentence.get_spans('ner'):
            if span.tag != 'O':
                all_entities.append({'text': span.text, 'type': span.tag})

    # Save entities to output file
    output_filename = os.path.splitext(filename)[0] + '_entities.txt'
    with open(os.path.join(output_dir, output_filename), 'w', encoding='utf-8') as f:
        entity_strings = ['{},{}'.format(ent['text'], ent['type']) for ent in all_entities]
        f.write('\n'.join(entity_strings))