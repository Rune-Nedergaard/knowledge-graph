"""
I forgot to include a tokenization step when creating the paragraphs.
In order to improve the embeddings for sematnic search, I will do that here now
"""

import os
from pathlib import Path
from danlp.models import load_bert_base_model
import torch
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_and_save_paragraphs(tokenizer, input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    for filename in tqdm(input_path.glob("*.txt"), total = len(list(input_path.glob("*.txt")))):
        with open(filename, encoding="utf-8") as f:
            paragraphs = f.read().split("\n")

        tokenized_paragraphs = []
        for paragraph in paragraphs:
            if paragraph.strip():
                tokenized_text = tokenizer.tokenize(paragraph)
                tokenized_paragraph = " ".join(tokenized_text)
                tokenized_paragraphs.append(tokenized_paragraph)

        output_filename = output_path / filename.name
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(tokenized_paragraphs))

if __name__ == "__main__":
    danish_bert = load_bert_base_model()


    tokenizer = danish_bert.tokenizer

    input_folder = "data/paragraphs"
    output_folder = "data/paragraphs_tokenized"

    os.makedirs(output_folder, exist_ok=True)

    tokenize_and_save_paragraphs(tokenizer, input_folder, output_folder)
