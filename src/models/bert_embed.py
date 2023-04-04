from torch import no_grad
import os, re
from typing import Union, List
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
import torch
import warnings

class BertEmbed:
    def __init__(self, model_path: str = None, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import BertTokenizer, BertModel
        import torch

        if model_path is not None:
            self.path_model = model_path
            self.pretrained = download_model('bert.botxo.pytorch', cache_dir, process_func=_unzip_process_func, verbose=verbose)
        else:
            # download model
            self.path_model = download_model('bert.botxo.pytorch', cache_dir, process_func=_unzip_process_func, verbose=verbose)

        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained)
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
        if len(tokenized_text) > 512:
            warnings.warn("The text is too long for BERT to handle. It will be truncated.")
            tokenized_text = tokenized_text[:512]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens]).to(self.model.device)  # Send tensor to the same device as the model
        segments_tensors = torch.tensor([segments_ids]).to(self.model.device)  # Send tensor to the same device as the model

        with no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        # sentence embedding
        sentence_embedding = torch.mean(hidden_states[-2][0], dim=0)

        return sentence_embedding
