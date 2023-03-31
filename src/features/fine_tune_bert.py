from danlp.models import load_bert_base_model
from torch.utils.data import Dataset

danish_bert = load_bert_base_model()


tokenizer = danish_bert.tokenizer
model = danish_bert.model

print(model)
print(tokenizer)