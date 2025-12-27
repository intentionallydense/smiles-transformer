import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from data import dataloader
import models.model


# Load real data
train_loader, val_loader, test_loader, tokenizer = dataloader.get_dataloaders(batch_size=8)

# Build model with real vocab size
model = model.Transformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    d_mlp=512,
    n_layers=4,
    max_len=128
)

# Grab a real batch
batch = next(iter(train_loader))
input_ids = batch['input_ids']
attention_mask = batch['attention_mask'].float()  # your model expects float, dataloader gives bool
targets = batch['target']

print(f"Input shape: {input_ids.shape}")
print(f"Mask shape: {attention_mask.shape}")
print(f"Target shape: {targets.shape}")

# Forward pass
output = model(input_ids, attention_mask)

print(f"Output shape: {output.shape}")  # should be (8, 1)
print(f"Predictions: {output.squeeze()}")
print(f"Actual targets: {targets}")
print(f"Any NaN? {t.isnan(output).any()}")

# Bonus: check that loss computes
loss_fn = nn.MSELoss()
loss = loss_fn(output.squeeze(), targets)
print(f"MSE Loss: {loss.item()}")

# Bonus: check backward pass works
loss.backward()
print("Backward pass succeeded!")

