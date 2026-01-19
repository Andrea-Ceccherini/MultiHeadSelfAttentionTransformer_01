import os
import sys
import csv
import random
from torch.utils import collect_env

# --- CRITICAL HARDWARE FIX FOR RX 9070 XT (RDNA 4) ---
# We force RDNA 3 compatibility. Since Phase 1 worked with this, we stick to it.
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.3"

# --- SYSTEM CONFIGURATION ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Debugging (Optional, can remove if too noisy)
os.environ["AMD_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file

from mh_sat_algorithms_for_custom_transformer_model import (
    TOKENIZATION_MAX_LENGTH, CustomTransformer,
    create_model_configuration, NUM_LAYERS, D_MODEL,
    NUM_HEADS, D_FF, DROPOUT, load_gpt2_tokenizer
)

# --- CONFIGURATION ---
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 1e-5


class BalancedFineTuningDataset(Dataset):
    def __init__(self, liver_csv_, wiki_file_, tokenizer_, max_len_):
        self.tokenizer = tokenizer_
        self.max_len = max_len_
        self.samples = []

        print("Loading Liver Data...")
        try:
            with open(liver_csv_, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        # EOS Token is Critical
                        text = f"Question: {row[0]} Answer: {row[1]}{self.tokenizer.eos_token}"
                        self.samples.append(text)
        except FileNotFoundError:
            print(f"❌ Error: {liver_csv_}")
            sys.exit(1)

        print("Loading Wikipedia Data...")
        liver_count = len(self.samples)
        wiki_samples = []
        try:
            with open(wiki_file_, 'r', encoding='utf-8', errors='replace') as f:
                buffer = ""
                for line in f:
                    line = line.strip()
                    if not line: continue
                    buffer += " " + line
                    if len(buffer.split()) > 50:
                        # EOS Token is Critical
                        wiki_samples.append(buffer.strip() + self.tokenizer.eos_token)
                        buffer = ""
                        if len(wiki_samples) >= liver_count * 2:
                            break
        except FileNotFoundError:
            print(f"❌ Error: {wiki_file_}")
            sys.exit(1)

        self.samples.extend(wiki_samples)
        random.shuffle(self.samples)
        print(f"Total Training Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encodings = self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        return {'input_ids': encodings['input_ids'].squeeze(0)}


def train_loop(epochs_, train_dl_, device_, optimizer_, criterion_, model_, save_dir_):
    print(f"Training on device: {device_}")

    # Move model
    model_.to(device_)
    model_.train()

    # Enable Mixed Precision (Matches Phase 1 Stability)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs_):
        total_loss = 0
        steps = 0
        optimizer_.zero_grad()

        print(f"\n--- Epoch {epoch + 1}/{epochs_} ---")

        for i, batch in enumerate(train_dl_):
            src_data = batch['input_ids'].to(device_)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            # Autocast (FP16) - This worked for your Phase 1
            with torch.autocast("cuda", dtype=torch.float16):
                output = model_(src_data, decoder_input)
                loss = criterion_(output.reshape(-1, output.shape[-1]), labels.reshape(-1))
                loss = loss / ACCUMULATION_STEPS

            # Scaled Backward
            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                # Gradient Clipping
                scaler.unscale_(optimizer_)
                torch.nn.utils.clip_grad_norm_(model_.parameters(), 1.0)

                scaler.step(optimizer_)
                scaler.update()
                optimizer_.zero_grad()

                current_loss = loss.item() * ACCUMULATION_STEPS
                total_loss += current_loss
                steps += 1

                if steps % 50 == 0:
                    print(f"\rStep {steps} | Loss: {current_loss:.4f}", end="")
                    # Clear cache to keep RDNA4 happy
                    torch.cuda.empty_cache()

        avg = total_loss / steps if steps > 0 else 0
        print(f"\n   Avg Loss: {avg:.4f}")

        os.makedirs(save_dir_, exist_ok=True)
        save_file(model_.state_dict(), os.path.join(save_dir_, "fine_tuned_best.safetensors"))


def print_hw_and_drivers_versions():
    print("\n--- RX 9070 XT Check ---")
    device_ = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device_}")
    print(f"HSA Override: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    print("------------------------\n")


def get_complete_system_info():
    """
    Collects standard PyTorch environment info plus specific
    environment variables relevant to AMD ROCm setup.
    """
    # 1. Get the standard report (The output you posted)
    raw_report = collect_env.get_pretty_env_info()

    # 2. Add custom check for your specific AMD/RDNA4 variables
    # This helps confirm if your script is actually seeing the overrides
    custom_vars = [
        "HSA_OVERRIDE_GFX_VERSION",  # Critical for RX 9070 XT
        "AMD_SERIALIZE_KERNEL",  # Critical for stability
        "PYTORCH_ALLOC_CONF",
        "LD_LIBRARY_PATH",
        "ROCM_PATH"
    ]

    extra_info = "\n\n---------- CUSTOM ENV VARIABLES CHECK ----------\n"
    for var in custom_vars:
        value = os.environ.get(var, "Not Set")
        extra_info += f"{var}: {value}\n"

    # 3. Add explicit GPU capability check
    gpu_check = "\n---------- PYTORCH INTERNAL GPU CHECK ----------\n"
    if torch.cuda.is_available():
        try:
            gpu_check += f"Is CUDA available: {torch.cuda.is_available()}\n"
            gpu_check += f"Device Name: {torch.cuda.get_device_name(0)}\n"
            gpu_check += f"Device Capability: {torch.cuda.get_device_capability(0)}\n"
        except Exception as e:
            gpu_check += f"Error querying GPU: {e}\n"
    else:
        gpu_check += "CUDA/ROCm not available in PyTorch.\n"

    return raw_report + extra_info + gpu_check


if __name__ == "__main__":
    print("MAIN (RDNA4 STABLE MODE) - BEGIN")

    print_hw_and_drivers_versions()
    report = get_complete_system_info()
    print("report =", report)

    # Auto-detect ROCm/CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"

    liver_csv = "../../../Datasets/LiverDataset/liver_questions_and_answers_999.csv"
    wiki_txt = "../../../Datasets/WikipediaDump/Final_Training_Data/train_chunk_001.txt"
    pretrained_path = os.path.join("../mh_sat_01_01/unsupervised_model_weights", "latest_checkpoint_27000.safetensors")
    save_dir = "../mh_sat_01_01/supervised_model_weights"

    tokenizer, vocab_size = load_gpt2_tokenizer()

    dataset = BalancedFineTuningDataset(liver_csv, wiki_txt, tokenizer, TOKENIZATION_MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loading Phase 1 Model...")
    model = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    if os.path.exists(pretrained_path):
        model.load_state_dict(load_file(pretrained_path))
        print("✅ Phase 1 Weights Loaded.")
    else:
        print(f"❌ Error: {pretrained_path} not found.")
        sys.exit(1)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    train_loop(EPOCHS, dataloader, device, optimizer, criterion, model, save_dir)

    create_model_configuration(save_dir, vocab_size, vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,
                               TOKENIZATION_MAX_LENGTH, DROPOUT)
    print("✅ Done.")