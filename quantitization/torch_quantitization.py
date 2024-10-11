"""
Dynamic Quantization: Suitable for models where you want a quick quantization without needing calibration data. This can be applied easily to linear layers.
Static Quantization: Requires calibration data to gather statistics on activations, providing potentially better performance than dynamic quantization.
Model Evaluation: Always evaluate the quantized model to ensure that it meets the performance requirements of your application.

"""


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased"  # You can choose any compatible model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()  # Set the model to evaluation model

# Apply dynamic quantization
# Apply dynamic quantization
quantized_model_dynamic = torch.quantization.quantize_dynamic(
    model,  # the original model
    {nn.Linear},  # layers to quantize
    dtype=torch.qint8  # quantization type
)

# Check the model size
print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Quantized model parameters: {sum(p.numel() for p in quantized_model_dynamic.parameters())}")


# Apply static quantization
# Step 1: Define a quantization configuration
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Step 2: Fuse layers (if applicable, e.g., Conv2d + ReLU)
# Here, we don't have convolutional layers, but if we did, we would use:
# model = torch.quantization.fuse_modules(model, [['fc1', 'relu']])

# Step 3: Prepare the model for static quantization
torch.quantization.prepare(model, inplace=True)

# Step 4: Calibrate the model with representative data
# Here, we will just create random input for demonstration
dummy_input = torch.randn(1, 784)
model(dummy_input)

# Step 5: Convert the model to a quantized version
quantized_model_static = torch.quantization.convert(model, inplace=True)

# Check the model size
print(f"Quantized static model parameters: {sum(p.numel() for p in quantized_model_static.parameters())}")

# Create a random input tensor
test_input = torch.randn(1, 784)

# Inference with dynamic quantized model
with torch.no_grad():
    output_dynamic = quantized_model_dynamic(test_input)
    print("Dynamic Quantized Output:", output_dynamic)

# Inference with static quantized model
with torch.no_grad():
    output_static = quantized_model_static(test_input)
    print("Static Quantized Output:", output_static)
