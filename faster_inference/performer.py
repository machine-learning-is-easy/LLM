"""
Performer uses kernelized attention to achieve linear time complexity for attention.
"""

from transformers import AutoModel, AutoTokenizer

# Load the existing model (e.g., BERT)
existing_model_name = "bert-base-uncased"
existing_model = AutoModel.from_pretrained(existing_model_name)

from transformers import PerformerModel, PerformerConfig

# Create a Performer configuration
performer_config = PerformerConfig(
    vocab_size=existing_model.config.vocab_size,
    hidden_size=existing_model.config.hidden_size,
    num_hidden_layers=existing_model.config.num_hidden_layers,
    num_attention_heads=existing_model.config.num_attention_heads,
    intermediate_size=existing_model.config.intermediate_size,
    hidden_act=existing_model.config.hidden_act,
    layer_norm_eps=existing_model.config.layer_norm_eps,
    max_position_embeddings=existing_model.config.max_position_embeddings,
    type_vocab_size=existing_model.config.type_vocab_size,
    # Add any other parameters required for Performer
)

# Initialize Performer model with the new configuration
performer_model = PerformerModel(performer_config)

import torch

# Define a mapping of layer names if necessary
for name, param in existing_model.named_parameters():
    if name in performer_model.state_dict():
        # Copy the weights
        performer_model.state_dict()[name].copy_(param.data)

# Set the Performer model to evaluation mode
performer_model.eval()

from transformers import PerformerTokenizer

# Load Performer tokenizer
tokenizer = PerformerTokenizer.from_pretrained("google/performer")

# Sample input to test the tokenizer
sample_input = tokenizer("This is a test input for Performer.", return_tensors="pt", padding=True, truncation=True)

# Forward pass with Performer
with torch.no_grad():
    outputs = performer_model(**sample_input)
    print(outputs.last_hidden_state)