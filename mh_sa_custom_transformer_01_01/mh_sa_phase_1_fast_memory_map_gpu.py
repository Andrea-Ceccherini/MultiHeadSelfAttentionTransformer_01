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

"""



import os
import sys
import numpy as np
from datetime import datetime

# --- CRITICAL HARDWARE FIX (RDNA 4) ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.3"

# --- SYSTEM CONFIGURATION ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Debug logging to verify compilation isn't frozen
os.environ["AMD_LOG_LEVEL"] = "3" 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

from mh_sa_algorithms_for_custom_transformer_model import (
    CustomTransformer, load_liver_tokenizer, TOKENIZATION_MAX_LENGTH, NUM_LAYERS, D_MODEL, D_FF, DROPOUT, NUM_HEADS
)

# --- CONFIGURATION ---
DATA_BIN_PATH = "../mh_sa_custom_transformer_01_01/bin_dataset/wiki_books_dataset.bin" # Adjusted to match your pre-processor output
CHECKPOINT_DIR = "../mh_sa_custom_transformer_01_01/unsupervised_model_weights"

# --- STABLE TRAINING PARAMS ---
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
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


def train_fast(epochs_, dataloader_, device_, optimizer_, criterion_, model_, save_dir_):
    print("train_fast() - BEGIN")
    print("Moving model to GPU (Compiling kernels... wait 2-5 mins)...")
    model_.to(device_)
    model_.train()
    print("✅ Model on GPU.")

    scaler = torch.amp.GradScaler("cuda")
    steps = 0
    total_loss = 0
    optimizer_.zero_grad(set_to_none=True)

    for epoch in range(epochs_):
        print(f"\n--- Epoch {epoch + 1}/{epochs_} ---")

        for i, batch in enumerate(dataloader_):
            src_data = batch['input_ids'].to(device_)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            with torch.autocast("cuda", dtype=torch.float16):
                output = model_(src_data, decoder_input)
                loss = criterion_(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

                if torch.isnan(loss):
                    print(f"\n⚠️ WARNING: NaN detected at batch {i}. Skipping update.")
                    optimizer_.zero_grad(set_to_none=True)
                    continue

                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            current_loss_val = loss.item() * ACCUMULATION_STEPS
            total_loss += current_loss_val

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer_)
                torch.nn.utils.clip_grad_norm_(model_.parameters(), 1.0)

                scaler.step(optimizer_)
                scaler.update()
                optimizer_.zero_grad(set_to_none=True)
                steps += 1

                if steps % 100 == 0:
                    avg = total_loss / (i + 1) # Note: 'i' is batches, not steps, but close enough for logging
                    print(f"\rStep {steps} | Loss: {current_loss_val:.4f}", end="", flush=True)

                if steps % 1000 == 0:
                    print(f"\n   💾 Saving Checkpoint at Step {steps}...")
                    os.makedirs(save_dir_, exist_ok=True)
                    save_file(model_.state_dict(), os.path.join(save_dir_, "latest_checkpoint.safetensors"))
                    torch.cuda.empty_cache()

        print(f"\nEpoch {epoch + 1} Complete.")
        torch.cuda.empty_cache()
    print("train_fast() - END")


if __name__ == "__main__":
    print("__main__() - BEGIN")
    begin_time = datetime.now()

    # 1. Verify Data Exists
    if not os.path.exists(DATA_BIN_PATH):
        print(f"❌ Error: {DATA_BIN_PATH} not found.")
        print("Did you run the preprocessor inside the correct folder?")
        sys.exit(1)

    # 2. Load Custom Tokenizer
    tokenizer, vocab_size = load_liver_tokenizer()

    model = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 3. Resume Logic (Fixed Paths)
    # We look in the same place we save to: CHECKPOINT_DIR
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.safetensors")

    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint: {checkpoint_path}")
        print("   Loading weights to RESUME training...")
        try:
            from safetensors.torch import load_file
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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train
    train_fast(1, dataloader, device, optimizer, criterion, model, CHECKPOINT_DIR)

    print(f"Elapsed: {datetime.now() - begin_time}")
    print("__main__() - END")