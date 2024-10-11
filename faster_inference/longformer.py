"""
Longformer uses a sliding window approach for self-attention, which allows it to handle longer sequences efficiently.
"""

from transformers import AutoModel, AutoTokenizer

# Load the existing model
existing_model_name = "bert-base-uncased"
existing_model = AutoModel.from_pretrained(existing_model_name)
from transformers import LongformerModel, LongformerConfig

# Create a Longformer configuration based on the existing model's configuration
longformer_config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")

# Initialize Longformer model with the new configuration
longformer_model = LongformerModel(longformer_config)


import torch

# Define a mapping of layer names if needed
# This example assumes compatible architectures, adjust as necessary
for name, param in existing_model.named_parameters():
    if name in longformer_model.state_dict():
        longformer_model.state_dict()[name].copy_(param.data)

# Ensure the Longformer model is set to evaluation mode
longformer_model.eval()

from transformers import LongformerTokenizer

# Load Longformer tokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# You can test the tokenizer with sample input
sample_input = tokenizer("This is a test input for Longformer.", return_tensors="pt", padding=True, truncation=True)

# Forward pass with Longformer
with torch.no_grad():
    outputs = longformer_model(**sample_input)
    print(outputs.last_hidden_state)

