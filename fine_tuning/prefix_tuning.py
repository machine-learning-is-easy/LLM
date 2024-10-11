"""
Prefix tuning is an efficient method for adapting pre-trained language models to specific tasks by prepending
learned embeddings (prefixes) to the input sequence. Here’s a step-by-step guide, along with code snippets,
to implement prefix tuning using the Hugging Face transformers library.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PrefixTuningModel(nn.Module):
    def __init__(self, model_name, prefix_length, hidden_size):
        super(PrefixTuningModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # Initialize prefix embeddings
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, hidden_size))

    def forward(self, input_ids, attention_mask=None):
        # Create the prefix tensor to concatenate with the input
        batch_size = input_ids.size(0)
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate prefix embeddings to input_ids
        input_embeddings = self.model.embeddings(input_ids)
        combined_embeddings = torch.cat((prefix, input_embeddings), dim=1)

        # Create a new attention mask to accommodate the prefix
        new_attention_mask = torch.cat(
            (torch.ones((batch_size, prefix.size(1)), device=input_ids.device), attention_mask), dim=1)

        # Forward pass through the model with the modified input
        outputs = self.model(inputs_embeds=combined_embeddings, attention_mask=new_attention_mask)

        return outputs


# Parameters
model_name = "gpt2"  # or any other compatible model
prefix_length = 5  # Length of the prefix
hidden_size = 768  # Adjust this based on the chosen model

# Initialize the Prefix Tuning model
prefix_tuning_model = PrefixTuningModel(model_name, prefix_length, hidden_size)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample input and corresponding labels (for language modeling)
text = "This is a sample input for prefix tuning."
inputs = tokenizer(text, return_tensors="pt")

# Prepare labels: for language modeling, labels are typically the same as input_ids, shifted one position to the right
labels = inputs['input_ids'].clone()  # Clone input_ids for labels
labels[:, :-1] = inputs['input_ids'][:, 1:]  # Shift labels to the right
labels[:, -1] = -100  # Ignore the last label during loss calculation

# Forward pass
with torch.no_grad():
    outputs = prefix_tuning_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

# Print output shape
print(outputs.last_hidden_state.shape)  # Output shape will vary based on the model

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Modify based on your task
optimizer = torch.optim.Adam(prefix_tuning_model.parameters(), lr=1e-5)

# Training loop (example)
num_epochs = 3  # Define number of epochs

for epoch in range(num_epochs):
    prefix_tuning_model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = prefix_tuning_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Calculate loss
    # We use `outputs.logits` because the outputs from the model include logits
    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item()}")