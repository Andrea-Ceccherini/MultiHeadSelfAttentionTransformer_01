"""
this script triggers saving data into three different types of caches.

    ~/.cache/amd/
    ~/.cache/miopen/
    ~/.config/miopen/



Run the following command before running this script

    # 1. Clear AMD Kernel Cache
    rm -rf ~/.cache/amd

    # 2. Clear MIOpen Cache (Math libraries)
    rm -rf ~/.cache/miopen
    rm -rf ~/.config/miopen

    # 3. Clear PyTorch Cache
    rm -rf ~/.cache/torch

    only if you change the structure of the work i.e.:

    If you change BATCH_SIZE from 4 to 8.
    If you change TOKENIZATION_MAX_LENGTH from 256 to 512.
    If you change the model size (e.g., NUM_LAYERS from 12 to 6).
    If you update your PyTorch version or GPU drivers.

this script get the data from file wiki_books_dataset.bin and create .safetensors file
CPU Time (Ryzen 7): Estimated 4 to 8 Weeks (running 24/7).
"""

import os
import sys
import numpy as np
from datetime import datetime
from tqdm import tqdm  # For progress bar

# --- FORCE CPU MODE ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

from mh_sa_algorithms_for_custom_transformer_model import (
    CustomTransformer, load_liver_tokenizer, TOKENIZATION_MAX_LENGTH, NUM_LAYERS, D_MODEL, D_FF, DROPOUT, NUM_HEADS
)

# --- CONFIGURATION ---
DATA_BIN_PATH = "bin_dataset/wiki_books_dataset.bin"
CHECKPOINT_DIR = "unsupervised_model_weights"  # Relative to script, if not in subfolder. Adjust if needed.

# --- STABLE TRAINING PARAMS (CPU) ---
BATCH_SIZE = 2  # Even smaller for CPU, to manage RAM
ACCUMULATION_STEPS = 16  # Effective Batch Size = 32 (Matches previous good runs)
LEARNING_RATE = 1e-4


class MemoryMapDataset(Dataset):
    def __init__(self, bin_path, block_size):
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        print(f"Dataset loaded. Total Tokens: {len(self.data):,}")

    def __len__(self):
        return len(self.data) // self.block_size - 1

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {'input_ids': chunk}


def train_fast_cpu(epochs_, dataloader_, device_, optimizer_, criterion_, model_, save_dir_):
    print("train_fast_cpu() - BEGIN")
    print(f"Training on device: {device_} (FORCED CPU)")
    model_.to(device_)
    model_.train()

    steps = 0
    total_loss = 0
    optimizer_.zero_grad(set_to_none=True)  # Good practice even for CPU

    for epoch in range(epochs_):
        print(f"\n--- Epoch {epoch + 1}/{epochs_} ---")

        # Progress bar
        progress_bar = tqdm(enumerate(dataloader_), total=len(dataloader_), desc=f"Epoch {epoch + 1}", unit="batch")

        for i, batch in progress_bar:
            src_data = batch['input_ids'].to(device_)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            # Standard Forward Pass (FP32, CPU) - No autocast, no scaler
            output = model_(src_data, decoder_input)
            loss = criterion_(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

            if torch.isnan(loss):
                print(f"\n⚠️ WARNING: NaN detected at batch {i}. Skipping update.")
                optimizer_.zero_grad(set_to_none=True)
                continue

            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model_.parameters(), 1.0)  # Still good for stability
                optimizer_.step()
                optimizer_.zero_grad(set_to_none=True)
                steps += 1

                current_loss = loss.item() * ACCUMULATION_STEPS
                total_loss += current_loss
                progress_bar.set_postfix(loss=f"{current_loss:.4f}", avg_loss=f"{(total_loss / steps):.4f}")

        avg = total_loss / steps if steps > 0 else 0
        print(f"\n✅ Epoch {epoch + 1} Finished. Avg Loss: {avg:.4f}")

        os.makedirs(save_dir_, exist_ok=True)
        save_file(model_.state_dict(), os.path.join(save_dir_, "latest_checkpoint.safetensors"))
        # No empty_cache for CPU

    print("train_fast_cpu() - END")


if __name__ == "__main__":
    print("__main__() - BEGIN")
    begin_time = datetime.now()

    # 1. Verify Data Exists
    if not os.path.exists(DATA_BIN_PATH):
        print(f"❌ Error: {DATA_BIN_PATH} not found. Run the preprocessor script first!")
        sys.exit(1)

    # 2. Load Custom Tokenizer
    tokenizer, vocab_size = load_liver_tokenizer()

    model = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    # FORCE CPU
    device = torch.device("cpu")
    print(f"Device: {device}")

    # 3. Resume Logic
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.safetensors")

    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint: {checkpoint_path}")
        print("   Loading weights to RESUME training...")
        try:
            model.load_state_dict(load_file(checkpoint_path))
            print("   ✅ Resume successful!")
        except Exception as e:
            print(f"   ⚠️ Error loading checkpoint: {e}")
            print("   Starting from scratch.")
    else:
        print(f"🆕 No checkpoint found at {checkpoint_path}. Starting fresh.")

    dataset = MemoryMapDataset(DATA_BIN_PATH, TOKENIZATION_MAX_LENGTH)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,  # Physical Batch Size (e.g., 2 sentences at a time)
        shuffle=True,  # Essential for good learning
        num_workers=4,  # Use multiple CPU cores for data loading
        pin_memory=True  # Speeds up CPU-to-CPU data transfer
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train
    train_fast_cpu(1, dataloader, device, optimizer, criterion, model, CHECKPOINT_DIR)

    print(f"Elapsed: {datetime.now() - begin_time}")
    print("__main__() - END")