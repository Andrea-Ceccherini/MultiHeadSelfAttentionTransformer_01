import os
import sys
import csv
import random
from tqdm import tqdm

# --- FORCE CPU MODE ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    model_.to(device_)
    model_.train()

    for epoch in range(epochs_):
        total_loss = 0
        steps = 0
        optimizer_.zero_grad()

        print(f"\n--- Epoch {epoch + 1}/{epochs_} ---")

        # CPU training needs a progress bar because it's slower per step
        progress_bar = tqdm(enumerate(train_dl_), total=len(train_dl_), desc=f"Epoch {epoch + 1}")

        for i, batch in progress_bar:
            src_data = batch['input_ids'].to(device_)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            output = model_(src_data, decoder_input)
            loss = criterion_(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model_.parameters(), 1.0)
                optimizer_.step()
                optimizer_.zero_grad()

                current_loss = loss.item() * ACCUMULATION_STEPS
                total_loss += current_loss
                steps += 1

                progress_bar.set_postfix(loss=f"{current_loss:.4f}")

        avg = total_loss / steps if steps > 0 else 0
        print(f"\n✅ Epoch {epoch + 1} Finished. Avg Loss: {avg:.4f}")

        os.makedirs(save_dir_, exist_ok=True)
        save_file(model_.state_dict(), os.path.join(save_dir_, "fine_tuned_best.safetensors"))


if __name__ == "__main__":
    print("MAIN (CPU SAFE MODE) - BEGIN")

    device = "cpu"

    liver_csv = "../../../Datasets/LiverDataset/liver_questions_and_answers_999.csv"
    wiki_txt = "../../../Datasets/WikipediaDump/Final_Training_Data/train_chunk_001.txt"
    # Point to your good Step 27k checkpoint
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