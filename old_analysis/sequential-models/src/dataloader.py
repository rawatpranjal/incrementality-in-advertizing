"""
PyTorch DataLoader utilities for the Causal Transformer model.

This module provides:
1. Custom Dataset class for session data
2. Collate function for batching variable-length sequences
3. Data loading utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import json
from pathlib import Path


class SessionDataset(Dataset):
    """
    PyTorch Dataset for session-based sequential data.

    Args:
        data_path: Path to parquet file containing session data
        vocab_path: Path to vocabulary JSON file
        max_sequence_length: Maximum sequence length (truncate if longer)
    """

    def __init__(
        self,
        data_path: str,
        vocab_path: str = None,
        max_sequence_length: int = 500
    ):
        # Load data - support both pickle and parquet
        if data_path.endswith('.pkl'):
            import pickle
            with open(data_path, 'rb') as f:
                self.df = pickle.load(f)
        else:
            self.df = pd.read_parquet(data_path)
        self.max_sequence_length = max_sequence_length

        # Load vocabulary if provided
        self.vocab = None
        if vocab_path:
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)

        print(f"Loaded {len(self.df):,} sessions from {Path(data_path).name}")

    def __len__(self) -> int:
        """Return number of sessions in dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single session.

        Returns:
            Dictionary containing:
            - event_types: List of event type IDs
            - item_ids: List of item IDs
            - time_deltas: List of time deltas in minutes
            - treatment: Binary treatment indicator
            - outcome: Binary outcome indicator
            - session_id: Session identifier
            - sequence_length: Actual sequence length
        """
        row = self.df.iloc[idx]

        # Extract sequence components
        sequence = row['sequence']

        # Truncate if necessary
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]

        # Unzip the sequence tuples
        if len(sequence) > 0:
            event_types, item_ids, time_deltas = zip(*sequence)
        else:
            event_types, item_ids, time_deltas = [], [], []

        return {
            'event_types': list(event_types),
            'item_ids': list(item_ids),
            'time_deltas': list(time_deltas),
            'treatment': row['treatment'],
            'outcome': row['outcome'],
            'session_id': row['session_id'],
            'sequence_length': len(sequence)
        }


def pad_sequence_batch(
    sequences: List[List],
    pad_value: Any = 0,
    dtype: torch.dtype = torch.long
) -> torch.Tensor:
    """
    Pad a batch of sequences to the same length.

    Args:
        sequences: List of sequences (each is a list)
        pad_value: Value to use for padding
        dtype: Data type for the tensor

    Returns:
        Padded tensor of shape (batch_size, max_length)
    """
    # Find maximum length in batch
    max_length = max(len(seq) for seq in sequences)

    # Create padded tensor
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_length), pad_value, dtype=dtype)

    # Fill in actual values
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            padded[i, :len(seq)] = torch.tensor(seq, dtype=dtype)

    return padded


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching session data.

    Handles variable-length sequences by padding them to the same length
    within each batch.

    Args:
        batch: List of dictionaries from SessionDataset

    Returns:
        Dictionary of batched tensors
    """
    # Extract components from batch
    event_types = [item['event_types'] for item in batch]
    item_ids = [item['item_ids'] for item in batch]
    time_deltas = [item['time_deltas'] for item in batch]
    treatments = torch.tensor([item['treatment'] for item in batch], dtype=torch.long)
    outcomes = torch.tensor([item['outcome'] for item in batch], dtype=torch.long)
    sequence_lengths = torch.tensor([item['sequence_length'] for item in batch], dtype=torch.long)
    session_ids = [item['session_id'] for item in batch]

    # Pad sequences
    event_types_padded = pad_sequence_batch(event_types, pad_value=0, dtype=torch.long)
    item_ids_padded = pad_sequence_batch(item_ids, pad_value=0, dtype=torch.long)
    time_deltas_padded = pad_sequence_batch(time_deltas, pad_value=0.0, dtype=torch.float32)

    return {
        'event_types': event_types_padded,
        'item_ids': item_ids_padded,
        'time_deltas': time_deltas_padded,
        'treatments': treatments,
        'outcomes': outcomes,
        'sequence_lengths': sequence_lengths,
        'session_ids': session_ids
    }


def create_data_loader(
    data_path: str,
    vocab_path: str = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_sequence_length: int = 500
) -> DataLoader:
    """
    Create a DataLoader for session data.

    Args:
        data_path: Path to parquet file
        vocab_path: Path to vocabulary JSON (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        max_sequence_length: Maximum sequence length

    Returns:
        PyTorch DataLoader
    """
    dataset = SessionDataset(
        data_path=data_path,
        vocab_path=vocab_path,
        max_sequence_length=max_sequence_length
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    return loader


def load_vocabularies(vocab_path: str) -> Dict[str, Any]:
    """
    Load vocabulary mappings from JSON file.

    Args:
        vocab_path: Path to vocabulary JSON file

    Returns:
        Dictionary containing vocabulary mappings
    """
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # Convert string keys back to integers where needed
    vocab['int_to_event'] = {int(k): v for k, v in vocab['int_to_event'].items()}
    vocab['int_to_item'] = {int(k): v for k, v in vocab['int_to_item'].items()}

    return vocab


class SessionBatchSampler:
    """
    Custom batch sampler that groups sessions by similar lengths.
    This can improve training efficiency by reducing padding.
    """

    def __init__(
        self,
        dataset: SessionDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Group sessions by length buckets
        self.length_indices = {}
        for idx in range(len(dataset)):
            length = dataset.df.iloc[idx]['sequence_length']
            # Create buckets of size 50 (0-49, 50-99, etc.)
            bucket = length // 50
            if bucket not in self.length_indices:
                self.length_indices[bucket] = []
            self.length_indices[bucket].append(idx)

    def __iter__(self):
        """Generate batches of indices."""
        # Collect all batches
        batches = []

        for bucket_indices in self.length_indices.values():
            # Shuffle within bucket if needed
            if self.shuffle:
                indices = np.random.permutation(bucket_indices).tolist()
            else:
                indices = bucket_indices.copy()

            # Create batches from bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Shuffle batches if needed
        if self.shuffle:
            np.random.shuffle(batches)

        # Yield batches
        for batch in batches:
            yield batch

    def __len__(self):
        """Return number of batches."""
        total = sum(len(indices) for indices in self.length_indices.values())
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size