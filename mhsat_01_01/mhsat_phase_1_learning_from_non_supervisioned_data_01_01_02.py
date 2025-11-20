import torch
from torch.utils.data import DataLoader, Dataset
import os
from safetensors.torch import save_file
from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    CustomTransformer, NUM_LAYERS, TOKENIZATION_MAX_LENGTH, 
    D_MODEL, NUM_HEADS, D_FF, DROPOUT, load_gpt2_tokenizer
)
from datetime import datetime

# Simple dataset for Unsupervised training
class UnsupervisedTextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

def calculate_elapsed_time(start, end):
    return str(end - start)

def model_training_unsupervised(epochs, dataloader, device, optimizer, criterion, model, model_save_dir, patience=3):
    print("Unsupervised Training - BEGIN")
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        
        print(f"Epoch {epoch+1}/{epochs} ", end="")
        
        for i, batch in enumerate(dataloader):
            src_data = batch['input_ids'].to(device)
            
            # Logic: Predict next token.
            # Encoder Input: Full sequence
            # Decoder Input: Full sequence minus last token
            # Target (Label): Full sequence minus first token
            
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]
            
            optimizer.zero_grad()
            output = model(src_data, decoder_input)
            
            loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if i % 50 == 0: print(".", end="", flush=True)

        avg_loss = train_loss / len(dataloader)
        print(f" Loss: {avg_loss:.4f}")

        # Simulating validation on the same set for simplicity (In real scenarios, split the data)
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            epochs_no_improve = 0
            os.makedirs(model_save_dir, exist_ok=True)
            save_file(model.state_dict(), os.path.join(model_save_dir, "unsupervised_model_best.safetensors"))
            print("Saved Best Model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
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
                    if len(text) > 10: # Skip empty files
                        data_list.append(text)
                        count += 1
            except Exception: pass
    return count

if __name__ == "__main__":
    print("MAIN - BEGIN")
    begin_time = datetime.now()

    # Folders
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Data
    raw_text = []
    for folder in folders:
        c = load_text_data_from_folder(folder, raw_text)
        print(f"Loaded {c} docs from {folder}")

    if not raw_text:
        print("No data found.")
        exit()

    print(f"Tokenizing {len(raw_text)} documents...")
    tokenized_data = tokenizer(
        raw_text, return_tensors='pt', padding='max_length', truncation=True, max_length=TOKENIZATION_MAX_LENGTH
    )

    dataset = UnsupervisedTextDataset(tokenized_data)
    # Increased batch size and workers
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model_training_unsupervised(5, dataloader, device, optimizer, criterion, model, "unsupervised_model_weights")

    print(f"Elapsed: {calculate_elapsed_time(begin_time, datetime.now())}")