import torch as t
import torch.nn as nn
from data.dataloader import get_dataloaders
from models.model import Transformer

device = t.device('cpu')

# Load data
train_loader, val_loader, test_loader, tokenizer = get_dataloaders(batch_size=64)

# Create model
model = Transformer(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_heads=4,
    d_mlp=512,
    n_layers=4,
    max_len=128
)

# Loss and optimizer
loss_fn =  nn.L1Loss()
optimizer = t.optim.AdamW(model.parameters(), lr=3e-3)

# Overfit check
'''
batch = next(iter(train_loader))
for i in range(500):
    predictions = model(batch['input_ids'].to(device), batch['attention_mask'].float().to(device))
    loss = loss_fn(predictions.squeeze(), batch['target'].to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"Step {i}: loss = {loss.item():.4f}")

'''
# Training loop
n_epochs = 20
best_val_loss = float('inf')

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids']
        mask = batch['attention_mask'].float()
        targets = batch['target']
        
        # Forward pass
        predictions = model(input_ids, mask)
        loss = loss_fn(predictions.squeeze(), targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if i % 100 == 0:
            print(f"batch {i}: loss = {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: train loss = {avg_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0
    with t.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            mask = batch['attention_mask'].float()
            targets = batch['target']

            predictions = model(input_ids, mask)
            loss = loss_fn(predictions.squeeze(), targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch: {epoch+1}, training loss: {avg_loss:.4f}, validation loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        t.save(model.state_dict(),'best_model.pt')
        print(f"new best average validation loss! {avg_val_loss:.4f} < {best_val_loss:.4f}")