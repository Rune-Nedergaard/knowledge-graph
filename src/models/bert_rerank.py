from torch import no_grad
import os, re
from typing import Union, List
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
import torch
import warnings

class BertRerank:
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

        # Load the fine-tuned model
        self.model = BertModel.from_pretrained(
            self.pretrained,
            state_dict=model_state_dict,
            output_hidden_states=True,  # Whether the model returns all hidden-states.
        )

    def embed_question_paragraph(self, question, paragraph):
        """
        Calculate the embeddings for the question and paragraph based on a BERT language model.
        """

        marked_text = "[CLS] " + question + " [SEP]" + paragraph + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        if len(tokenized_text) > 512:
            warnings.warn("The text is too long for BERT to handle. It will be truncated.")
            tokenized_text = tokenized_text[:512]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [0] * len(tokenized_text)
        sep_idx = tokenized_text.index('[SEP]')
        for i in range(sep_idx + 1, len(tokenized_text)):
            segments_ids[i] = 1

        tokens_tensor = torch.tensor([indexed_tokens]).to(self.model.device)  # Send tensor to the same device as the model
        segments_tensors = torch.tensor([segments_ids]).to(self.model.device)  # Send tensor to the same device as the model

        with no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        # embeddings
        question_embedding = torch.mean(hidden_states[-2][0, 1:sep_idx], dim=0)
        paragraph_embedding = torch.mean(hidden_states[-2][0, sep_idx+1:-1], dim=0)

        return question_embedding, paragraph_embedding

    def predict_similarity(self, question, paragraph):
        """
        Calculate the similarity between a question and a paragraph using their embeddings.
        """
        question_embedding, paragraph_embedding = self.embed_question_paragraph(question, paragraph)
        similarity = torch.cosine_similarity(question_embedding.unsqueeze(0), paragraph_embedding.unsqueeze(0)).item()
        return similarity
    

    """
    BATCH PROCESSING DOES NOT SEEM TO WORK    
    """
    def embed_question_paragraph_batch(self, question, paragraphs):
        marked_texts = ["[CLS] " + question + " [SEP]" + paragraph + " [SEP]" for paragraph in paragraphs]
        tokenized_texts = [self.tokenizer.tokenize(marked_text) for marked_text in marked_texts]

        # Truncate long texts
        tokenized_texts = [t[:512] if len(t) > 512 else t for t in tokenized_texts]
        
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(tokenized_text) for tokenized_text in tokenized_texts]
        max_len = max(len(tokens) for tokens in indexed_tokens)

        # Padding
        padded_tokens = [tokens + [0] * (max_len - len(tokens)) for tokens in indexed_tokens]

        segment_ids = []
        for tokens in padded_tokens:
            sep_idx = tokens.index(self.tokenizer.sep_token_id)
            segments = [0] * len(tokens)
            for i in range(sep_idx + 1, len(tokens)):
                segments[i] = 1
            segment_ids.append(segments)

        tokens_tensor = torch.tensor(padded_tokens).to(self.model.device)
        segments_tensors = torch.tensor(segment_ids).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        question_embeddings = []
        paragraph_embeddings = []

        for i, hidden_state in enumerate(hidden_states[-2]):
            sep_idx = indexed_tokens[i].index(self.tokenizer.sep_token_id)
            question_emb = torch.mean(hidden_state[1:sep_idx], dim=0)
            paragraph_emb = torch.mean(hidden_state[sep_idx+1:-1], dim=0)

            question_embeddings.append(question_emb)
            paragraph_embeddings.append(paragraph_emb)

        return question_embeddings, paragraph_embeddings

    def predict_similarity_batch(self, question, paragraphs):
        question_embeddings, paragraph_embeddings = self.embed_question_paragraph_batch(question, paragraphs)
        question_embeddings = torch.stack(question_embeddings)
        paragraph_embeddings = torch.stack(paragraph_embeddings)

        similarities = torch.cosine_similarity(question_embeddings, paragraph_embeddings).tolist()
        return similarities


