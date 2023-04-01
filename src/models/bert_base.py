from torch import no_grad
import os, re
from typing import Union, List
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
import torch
import warnings

class BertBase:
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import BertTokenizer, BertModel
        import torch
        # download model
        self.path_model= download_model('bert.botxo.pytorch', cache_dir, process_func=_unzip_process_func,verbose=verbose)
        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.path_model)
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained(self.path_model,
                                          output_hidden_states = True, # Whether the model returns all hidden-states.
                                          )


    def embed_text(self, text):
        """
        Calculate the embeddings for the sentence based on a BERT language model.
        """

        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        # sentence embedding
        sentence_embedding = torch.mean(hidden_states[-2][0], dim=0)

        return sentence_embedding
