

import torch
import torch.nn as nn
import torch.optim as optim

class DynamicPruningSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, pruning_threshold=0.1):
        super(DynamicPruningSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.pruning_threshold = pruning_threshold

        # Query, Key, and Value matrices
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Fully connected output
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads different pieces
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = queries.view(N, query_len, self.heads, self.head_dim)

        # Apply dynamic pruning before calculating attention
        queries = self.dynamic_prune(self.queries(queries), self.pruning_threshold)
        keys = self.dynamic_prune(self.keys(keys), self.pruning_threshold)
        values = self.dynamic_prune(self.values(values), self.pruning_threshold)

        # Attention calculation: Query * Key^T / sqrt(d_k)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # Attention * Values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)

        return self.fc_out(out)

    def dynamic_prune(self, matrix, threshold):
        """
        Dynamically prune the weights of the query, key, and value matrices
        based on the magnitude of the elements. Elements with an absolute
        value below the threshold are set to zero.
        """
        mask = matrix.abs() >= threshold
        pruned_matrix = matrix * mask.float()
        return pruned_matrix

    def train_dynamic_pruning_attention(model, optimizer, criterion, dataloader, num_epochs=50):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in dataloader:
                inputs, targets = batch
                optimizer.zero_grad()

                # Forward pass through self-attention
                output = model(inputs, inputs, inputs, mask=None)
                loss = criterion(output, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Optionally, adjust the pruning threshold during training
            if epoch % 10 == 0:
                model.pruning_threshold *= 0.95  # Reduce threshold for more aggressive pruning

            print(f"Epoch [{epoch}/{num_epochs}], Loss: {running_loss:.4f}")

    # Example usage
    embed_size = 256
    heads = 8
    attention_model = DynamicPruningSelfAttention(embed_size, heads)

    optimizer = optim.Adam(attention_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Assuming dataloader is predefined with (input, target) pairs
    train_dynamic_pruning_attention(attention_model, optimizer, criterion, dataloader)