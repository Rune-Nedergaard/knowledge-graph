from torch import no_grad
import os, re
from typing import Union, List
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
import torch
import warnings
from transformers import BertTokenizer, BertModel, BertConfig


class BertEmbed:
    def __init__(self, model_path: str = None, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import BertTokenizer, BertModel
        import torch

        # download pretrained model
        self.pretrained = download_model('bert.botxo.pytorch', cache_dir, process_func=_unzip_process_func, verbose=verbose)

        if model_path is not None:
            self.path_model = model_path
        else:
            self.path_model = self.pretrained

        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained)

        # Load the pre-trained model (weights)
        model_state_dict = torch.load(self.path_model)

        # Remove 'bert_model.' prefix from the fine-tuned model state dictionary keys
        model_state_dict = {k.replace('bert_model.', ''): v for k, v in model_state_dict.items()}

        # Remove the additional linear layer keys to make the architecture match the pretrained model
        model_state_dict.pop('linear.bias', None)
        model_state_dict.pop('linear.weight', None)


        # Load the fine-tuned model
        self.model = BertModel.from_pretrained(
            self.pretrained,
            state_dict=model_state_dict,
            output_hidden_states=True,  # Whether the model returns all hidden-states.
        )


        """This is for debugging purposes only, but it seems to work fine now"""
        # # Load a pretrained model for checking
        # pretrained_model = BertModel.from_pretrained(self.pretrained, output_hidden_states=True)

        # # Check if keys and parameter shapes are consistent between pretrained and fine-tuned model state dictionaries
        # pretrained_keys = set(pretrained_model.state_dict().keys())
        # fine_tuned_keys = set(model_state_dict.keys())

        # sorted_pretrained_keys = sorted(list(pretrained_keys))
        # sorted_fine_tuned_keys = sorted(list(fine_tuned_keys))

        # if sorted_pretrained_keys != sorted_fine_tuned_keys:
        #     raise ValueError("The keys in the pretrained and fine-tuned model state dictionaries do not match.")

        # for key in pretrained_keys:
        #     pretrained_shape = tuple(pretrained_model.state_dict()[key].shape)
        #     fine_tuned_shape = tuple(model_state_dict[key].shape)

        #     if pretrained_shape != fine_tuned_shape:
        #         raise ValueError(f"Parameter shapes do not match for key: {key}\n"
        #                         f"Pretrained shape: {pretrained_shape}, Fine-tuned shape: {fine_tuned_shape}")


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
