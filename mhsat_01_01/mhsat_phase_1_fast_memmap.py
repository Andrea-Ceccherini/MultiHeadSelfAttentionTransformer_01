import os
import sys
import numpy as np
from datetime import datetime

# --- SYSTEM CONFIGURATION ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    CustomTransformer, load_gpt2_tokenizer
)

# --- CONFIGURATION ---
DATA_BIN_PATH = "wiki_books_dataset.bin"
TOKENIZATION_MAX_LENGTH = 256
NUM_LAYERS = 12
D_MODEL = 768
NUM_HEADS = 12
D_FF = 3072
DROPOUT = 0.2

# --- STABLE TRAINING PARAMS ---
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4


# --- MEMORY MAPPED DATASET ---
class MemmapDataset(Dataset):
    def __init__(self, bin_path, block_size):
        self.block_size = block_size

        # Load the binary file as a memory-mapped array
        # This is INSTANT and uses almost 0 RAM (OS handles paging)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        print(f"Dataset loaded. Total Tokens: {len(self.data):,}")

    def __len__(self):
        # Number of possible chunks
        return len(self.data) // self.block_size - 1

    def __getitem__(self, idx):
        # Grab a chunk of data
        # We start at idx * block_size
        # But for better randomness, usually we do random sampling.
        # Since DataLoader shuffles 'idx', this is fine.
        start = idx * self.block_size
        end = start + self.block_size

        # Convert uint16 back to long for PyTorch
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {'input_ids': chunk}


def train_fast(epochs, dataloader, device, optimizer, criterion, model, save_dir):
    print("--- Phase 1 Training (Fast Memmap Mode) BEGIN ---")
    model.to(device)
    model.train()

    scaler = torch.amp.GradScaler("cuda")
    steps = 0
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        # Enumerate gives us a progress bar naturally
        for i, batch in enumerate(dataloader):
            src_data = batch['input_ids'].to(device)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            with torch.autocast("cuda", dtype=torch.float16):
                output = model(src_data, decoder_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            current_loss_val = loss.item() * ACCUMULATION_STEPS
            total_loss += current_loss_val

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1

                if steps % 100 == 0:
                    avg = total_loss / (i + 1)
                    print(f"\rStep {steps} | Loss: {current_loss_val:.4f} | Avg: {avg:.4f}", end="", flush=True)

                if steps % 1000 == 0:
                    print(f"\n   💾 Saving Checkpoint at Step {steps}...")
                    os.makedirs(save_dir, exist_ok=True)
                    save_file(model.state_dict(), os.path.join(save_dir, "latest_checkpoint.safetensors"))
                    torch.cuda.empty_cache()

        print(f"\nEpoch {epoch + 1} Complete.")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    begin_time = datetime.now()

    if not os.path.exists(DATA_BIN_PATH):
        print(f"❌ Error: {DATA_BIN_PATH} not found. Run the preprocessor script first!")
        sys.exit(1)

    tokenizer, vocab_size = load_gpt2_tokenizer()

    model = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset (Instant Load)
    dataset = MemmapDataset(DATA_BIN_PATH, TOKENIZATION_MAX_LENGTH)

    # DataLoader (Standard Map-Style allows SHUFFLE=TRUE which is better for learning)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # This drastically improves training quality vs streaming
        num_workers=4,  # You can try increasing this to 4 or 8 since we read from binary
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train
    train_fast(1, dataloader, device, optimizer, criterion, model, "unsupervised_model_weights")

    print(f"Elapsed: {datetime.now() - begin_time}")