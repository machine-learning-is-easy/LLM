import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim

# Example CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel()

# Prune 50% of connections in conv1 and conv2 layers
prune.l1_unstructured(model.conv1, name="weight", amount=0.5)
prune.l1_unstructured(model.conv2, name="weight", amount=0.5)
# Check sparsity (percentage of zero weights) in conv1 layer
def check_sparsity(layer):
    total_params = layer.weight.nelement()
    zero_params = torch.sum(layer.weight == 0)
    sparsity = 100.0 * zero_params / total_params
    return sparsity.item()

conv1_sparsity = check_sparsity(model.conv1)
conv2_sparsity = check_sparsity(model.conv2)

print(f"Sparsity in conv1: {conv1_sparsity:.2f}%")
print(f"Sparsity in conv2: {conv2_sparsity:.2f}%")

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
def train(model, optimizer, criterion, data_loader, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Example data loader (using random data)
dummy_loader = [(torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))) for _ in range(100)]
train(model, optimizer, criterion, dummy_loader)

# Remove pruning, making the pruning permanent
prune.remove(model.conv1, 'weight')
prune.remove(model.conv2, 'weight')

torch.save(model.state_dict(), "pruned_model.pth")