
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the pre-trained model and tokenizer from Hugging Face
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prune 30% of the weights in all the linear layers of the model
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # Apply L1 unstructured pruning on linear layers
        prune.l1_unstructured(module, name="weight", amount=0.3)

# Function to calculate sparsity
def check_sparsity(layer):
    total_params = layer.weight.nelement()
    zero_params = torch.sum(layer.weight == 0)
    sparsity = 100.0 * zero_params / total_params
    return sparsity.item()

# Check sparsity in pruned layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        sparsity = check_sparsity(module)
        print(f"Sparsity in {name}: {sparsity:.2f}%")

# Example input for evaluation
inputs = tokenizer("Hugging Face models are great!", return_tensors="pt")

# Forward pass (inference)
outputs = model(**inputs)
logits = outputs.logits
print(logits)

# Remove pruning from the model to make it permanent
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, 'weight')

# Save the pruned model
model.save_pretrained("pruned_bert_model")
tokenizer.save_pretrained("pruned_bert_model")

