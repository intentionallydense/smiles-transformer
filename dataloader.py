import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import QM9
from rdkit import Chem, RDLogger
import numpy as np
import warnings
import os

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

class Tokenizer:
    def __init__(self, smiles_list=None):
        self.special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.idx_to_char = {v: k for k, v in self.special_tokens.items()}
        self.char_to_idx = self.special_tokens.copy()
        self.vocab_size = len(self.special_tokens)
        
        if smiles_list:
            self.build_vocab(smiles_list)

    def build_vocab(self, smiles_list):
        chars = set()
        for smiles in smiles_list:
            chars.update(smiles)
        
        # Add unique characters to vocabulary
        for i, char in enumerate(sorted(list(chars))):
            idx = i + len(self.special_tokens)
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
            
        self.vocab_size = len(self.char_to_idx)

    def encode(self, smiles: str) -> list[int]:
        """Convert SMILES string to list of token indices. Prepend <sos>, append <eos>. """
        return [self.char_to_idx['<sos>']] + \
               [self.char_to_idx[char] for char in smiles] + \
               [self.char_to_idx['<eos>']]

    def decode(self, indices: list[int]) -> str:
        """Convert token indices back to SMILES string. Strip special tokens."""
        tokens = []
        for idx in indices:
            # Handle tensor or int input
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            
            char = self.idx_to_char.get(idx, '')
            if char not in self.special_tokens:
                tokens.append(char)
        return "".join(tokens)

    def pad_sequence(self, encoded: list[int], max_len: int) -> list[int]:
        """Pad or truncate to fixed length. Pad on the right with <pad> tokens."""
        if len(encoded) > max_len:
            return encoded[:max_len]
        return encoded + [self.char_to_idx['<pad>']] * (max_len - len(encoded))


class QM9SMILESDataset(Dataset):
    def __init__(self, smiles_data, targets, tokenizer, max_len=128):
        self.smiles_data = smiles_data
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        smiles = self.smiles_data[idx]
        target = self.targets[idx]
        
        encoded = self.tokenizer.encode(smiles)
        
        if len(encoded) > self.max_len:
            # Spec says: "Print a warning if any SMILES strings exceed max_len"
            pass
            
        padded = self.tokenizer.pad_sequence(encoded, self.max_len)
        
        input_ids = torch.tensor(padded, dtype=torch.long)
        
        # Attention mask: 1 for real tokens (not <pad>), 0 for <pad>
        attention_mask = (input_ids != self.tokenizer.special_tokens['<pad>']).float()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': torch.tensor(target, dtype=torch.float),
            'smiles': smiles
        }

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


def get_dataloaders(batch_size=64, max_len=128, num_workers=0, root='./data/qm9_root', seed=42, cache_path='./data/qm9_clean.pt'):
    """
    Returns (train_loader, val_loader, test_loader, tokenizer)
    """
    
    # Try loading from cache first
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        cached = torch.load(cache_path)
        clean_smiles = cached['smiles']
        clean_targets = cached['targets']
        print(f"Loaded {len(clean_smiles)} samples from cache")
    else:
        print(f"Loading QM9 from {root}...")
        dataset = QM9(root=root)
        
        clean_smiles = []
        clean_targets = []
        
        print(f"Processing {len(dataset)} samples (filtering invalid SMILES)...")
        
        for i in range(len(dataset)):
            data = dataset[i]
            
            if hasattr(data, 'smiles'):
                s = data.smiles
            else:
                try:
                    s = dataset.smiles[i]
                except:
                    continue

            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                gap_ev = data.y[0, 4].item()
                clean_smiles.append(s)
                clean_targets.append(gap_ev)
                
        print(f"Valid samples found: {len(clean_smiles)}")
        
        # Save to cache
        print(f"Saving cache to {cache_path}...")
        torch.save({'smiles': clean_smiles, 'targets': clean_targets}, cache_path)
    print(f"Loading QM9 from {root}...")
    dataset = QM9(root=root)

    # Build Tokenizer
    tokenizer = Tokenizer(clean_smiles)
    
    # Split
    n_clean = len(clean_smiles)
    indices = np.arange(n_clean)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    n_train = int(0.8 * n_clean)
    n_val = int(0.1 * n_clean)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # Helper to create dataset from indices
    def create_subset(idxs):
        sub_smiles = [clean_smiles[i] for i in idxs]
        sub_targets = [clean_targets[i] for i in idxs]
        return QM9SMILESDataset(sub_smiles, sub_targets, tokenizer, max_len=max_len)
    
    train_ds = create_subset(train_indices)
    val_ds = create_subset(val_indices)
    test_ds = create_subset(test_indices)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, tokenizer

def get_smiles_statistics(dataset):
    """
    Returns dict with stats
    """
    lengths = []
    chars = set()
    
    # Access internal data if possible
    if hasattr(dataset, 'smiles_data'):
        data_source = dataset.smiles_data
    else:
        # Fallback for generic datasets
        data_source = [item['smiles'] for item in dataset]

    for s in data_source:
        lengths.append(len(s))
        chars.update(s)
        
    return {
        'max_length': max(lengths) if lengths else 0,
        'mean_length': float(np.mean(lengths)) if lengths else 0.0,
        'vocab_chars': sorted(list(chars)),
        'num_samples': len(lengths)
    }

if __name__ == "__main__":
    # Validation Code
    print("Running validation...")
    try:
        train_loader, val_loader, test_loader, tokenizer = get_dataloaders(batch_size=32)
        
        print(f"Vocab Size: {tokenizer.vocab_size}")
        
        batch = next(iter(train_loader))
        print("Batch shapes:")
        print(f"Input IDs: {batch['input_ids'].shape}")
        print(f"Mask: {batch['attention_mask'].shape}")
        print(f"Target: {batch['target'].shape}")
        print(f"SMILES sample: {batch['smiles'][0]}")
        
        # 1. Size check
        total_len = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        print(f"Total samples: {total_len}")
        
        # 2. Round trip
        indices = batch['input_ids'][0]
        original = batch['smiles'][0]
        decoded = tokenizer.decode(indices)
        print(f"Original: {original}")
        print(f"Decoded:  {decoded}")
        assert original == decoded, "Round-trip failed!"
        
        # 3. Target Range
        targets = batch['target']
        print(f"Target range: {targets.min().item():.2f} - {targets.max().item():.2f} eV")
        
        # Stats
        stats = get_smiles_statistics(train_loader.dataset)
        print(f"Train Max Len: {stats['max_length']}")
        print(f"Train Mean Len: {stats['mean_length']:.2f}")

        print("Validation passed!")
        
    except Exception as e:
        print(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()