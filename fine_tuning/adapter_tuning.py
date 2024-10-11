"""
Key Points
Adapter Layers: The adapter layers allow you to fine-tune a pre-trained model without modifying the original weights,
which saves memory and computational resources.
Activation Functions: You can customize the activation function for the adapter layers through the AdapterConfig.
Parameter Efficiency: By only training the adapter parameters, you keep the majority of the model's parameters frozen,
making the training process faster and requiring less data.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import AdapterConfig, AdapterType


class AdapterTuningModel(nn.Module):
    def __init__(self, model_name):
        super(AdapterTuningModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # Add a new adapter
        self.adapter_config = AdapterConfig(
            hidden_size=self.model.config.hidden_size,
            adapter_size=64,  # Size of the adapter layer
            activation_function="relu",
        )

        # Register the adapter
        self.model.add_adapter("my_adapter", self.adapter_config)
        self.model.set_active_adapters("my_adapter")  # Set the active adapter

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs


# Parameters
model_name = "gpt2"  # or any other compatible model

# Initialize the Adapter Tuning model
adapter_tuning_model = AdapterTuningModel(model_name)


tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample input
text = "This is a sample input for adapter tuning."
inputs = tokenizer(text, return_tensors="pt")


# Forward pass
with torch.no_grad():
    outputs = adapter_tuning_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

# Print output shape
print(outputs.last_hidden_state.shape)  # Output shape will vary based on the model

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Modify based on your task
optimizer = torch.optim.Adam(adapter_tuning_model.parameters(), lr=1e-5)

# Training loop (example)
num_epochs = 3  # Define number of epochs

# Sample labels (for language modeling, the labels can be the same as input_ids shifted one position)
labels = inputs['input_ids'].clone()  # Clone input_ids for labels
labels[:, :-1] = inputs['input_ids'][:, 1:]  # Shift labels to the right
labels[:, -1] = -100  # Ignore the last label during loss calculation

for epoch in range(num_epochs):
    adapter_tuning_model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = adapter_tuning_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Calculate loss
    loss = criterion(outputs.last_hidden_state.view(-1, outputs.last_hidden_state.size(-1)), labels.view(-1))
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item()}")


