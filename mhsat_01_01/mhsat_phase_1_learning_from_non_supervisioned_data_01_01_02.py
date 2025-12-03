import os
import sys
from datetime import datetime

# --- SYSTEM CONFIGURATION & STABILITY FIXES ---
# 1. Prevent tokenizer deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. Memory Allocator Fix for RDNA 4
# We use the NEW variable name (PYTORCH_ALLOC_CONF) to avoid the deprecation warning.
# 'max_split_size_mb:128' reduces memory fragmentation which causes the Error 700 crash.
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"

# 3. Clean RDNA 4 Environment
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Silence logs
import logging
import torch._inductor.config

torch._inductor.config.verbose_progress = False
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file

# Import architecture
from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    CustomTransformer, load_gpt2_tokenizer
)

# --- MODEL CONFIGURATION (Large) ---
TOKENIZATION_MAX_LENGTH = 256
NUM_LAYERS = 12
D_MODEL = 768
NUM_HEADS = 12
D_FF = 3072
DROPOUT = 0.2


class UnsupervisedTextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


def calculate_elapsed_time(start, end):
    return str(end - start)


def print_model_stats(model, vocab_size):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total_params * 4) / (1024 * 1024)

    print("\n" + "=" * 50)
    print(f"📊 MODEL CONFIGURATION & STATISTICS")
    print("=" * 50)
    print(f"Architecture     : CustomTransformer (Decoder-Only)")
    print(f"Vocabulary Size  : {vocab_size}")
    print(f"Context Window   : {TOKENIZATION_MAX_LENGTH}")
    print("-" * 50)
    print(f"Layers (Depth)   : {NUM_LAYERS}")
    print(f"Embedding Dim    : {D_MODEL}")
    print(f"Total Parameters : {total_params:,}")
    print(f"Est. Model Size  : {size_mb:.2f} MB (Weights only)")
    print("=" * 50 + "\n")


def model_training_unsupervised(epochs, dataloader_, device_, optimizer_, criterion_, model_, model_save_dir_,
                                patience_=3):
    print("Unsupervised Training - BEGIN")

    model_.to(device_)

    # --- CRITICAL FIX: DISABLED COMPILATION ---
    # Your logs proved that torch.compile is causing "Illegal Memory Access" (Error 700)
    # on this specific large model. We MUST run in standard Eager mode for stability.
    print("✅ Running in Native Eager Mode (Stable)...")
    # model_ = torch.compile(model_) # DISABLED

    model_.train()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        train_loss = 0
        model_.train()

        print(f"Epoch {epoch + 1}/{epochs} ", end="")

        for i, batch in enumerate(dataloader_):
            src_data = batch['input_ids'].to(device_)

            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            optimizer_.zero_grad(set_to_none=True)

            # Autocast (FP16)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model_(src_data, decoder_input)
                loss = criterion_(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

            # Scaled Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer_)
            scaler.update()

            train_loss += loss.item()
            if i % 50 == 0: print(".", end="", flush=True)

        avg_loss = train_loss / len(dataloader_)
        print(f" Loss: {avg_loss:.4f}")

        # --- FIX: CLEAR CACHE ---
        # Important for preventing VRAM fragmentation between epochs
        torch.cuda.empty_cache()

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            epochs_no_improve = 0
            os.makedirs(model_save_dir_, exist_ok=True)
            save_file(model_.state_dict(), os.path.join(model_save_dir_, "unsupervised_model_best.safetensors"))
            print("   [Checkpoint] Saved Best Model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_:
                print("Early Stopping.")
                break


def load_text_data_from_folder(folder_path, data_list):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return 0
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(folder_path, filename), 'r', encoding='latin-1') as f:
                    text = f.read().strip()
                    if len(text) > 10:
                        data_list.append(text)
                        count += 1
            except Exception:
                pass
    return count


if __name__ == "__main__":
    print("MAIN - BEGIN")
    begin_time = datetime.now()

    folders = [
        "../../../Datasets/Txt_Books/",
        "../../../Datasets/WikipediaData/",
        "../../../Datasets/Au_Books/"
    ]

    tokenizer, vocab_size = load_gpt2_tokenizer()

    model = CustomTransformer(
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=TOKENIZATION_MAX_LENGTH,
        dropout=DROPOUT
    )

    print_model_stats(model, vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    raw_text = []
    for folder in folders:
        c = load_text_data_from_folder(folder, raw_text)
        print(f"Loaded {c} docs from {folder}")

    if not raw_text:
        print("No data found. Creating dummy data.")
        raw_text = ["This is a test sentence for the GPU." for _ in range(1000)]

    print(f"Tokenizing {len(raw_text)} documents...")
    tokenized_data = tokenizer(
        raw_text, return_tensors='pt', padding='max_length', truncation=True, max_length=TOKENIZATION_MAX_LENGTH
    )

    dataset = UnsupervisedTextDataset(tokenized_data)

    # --- CRITICAL FIX: BATCH SIZE 8 ---
    # 12 was too high and caused the crash. 8 is safe for this model size.
    batch_size = 8
    print(f"DataLoader Config: Batch Size={batch_size}, Workers=0, Pin Memory=False")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model_training_unsupervised(5, dataloader, device, optimizer, criterion, model, "unsupervised_model_weights")

    print(f"Elapsed: {calculate_elapsed_time(begin_time, datetime.now())}")
    print("MAIN - END")