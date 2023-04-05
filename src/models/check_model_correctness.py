import torch
from transformers import BertTokenizer, BertModel
from danlp.models import load_bert_base_model
from danlp.download import DEFAULT_CACHE_DIR, download_model, _unzip_process_func
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.bert_embed import BertEmbed

# Load the fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/fine_tuned_model.pth"
model = BertEmbed(model_path=model_path)
fine_tuned_model = model.model.to(device)

# Load the pretrained model
pretrained_model = load_bert_base_model()
pretrained_model = pretrained_model.model.to(device)



# Print the keys of the fine-tuned model state dict
print("Fine-tuned model keys:")
print(list(fine_tuned_model.state_dict().keys()))

# Print the keys of the pretrained model state dict
print("\nPretrained model keys:")
print(list(pretrained_model.state_dict().keys()))

# Compare the keys
fine_tuned_keys = set(fine_tuned_model.state_dict().keys())
pretrained_keys = set(pretrained_model.state_dict().keys())
print("\nKeys in fine-tuned but not in pretrained:")
print(list(fine_tuned_keys - pretrained_keys))
print("\nKeys in pretrained but not in fine-tuned:")
print(list(pretrained_keys - fine_tuned_keys))

# Compare the parameters
print("\nComparing parameters:")
for key in fine_tuned_keys.intersection(pretrained_keys):
    fine_tuned_param = fine_tuned_model.state_dict()[key]
    pretrained_param = pretrained_model.state_dict()[key]
    if not torch.allclose(fine_tuned_param, pretrained_param):
        print(f"Parameters differ for key: {key}")
    else:
        print(f"Parameters are the same for key: {key}")


print("hi)")