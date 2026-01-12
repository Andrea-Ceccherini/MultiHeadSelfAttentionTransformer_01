import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import os
import glob
import random

# ------- Global Parameters -------
TOKENIZATION_MAX_LENGTH = 256   # To prevent answer cut off. Gives the model more "runway" allows it to read longer Wikipedia contexts in Phase 1, making it smarter at constructing sentences.
NUM_LAYERS = 12
D_MODEL = 768
NUM_HEADS = 12
D_FF = 3072
DROPOUT = 0.2   # In Transformer theory, the Feed Forward layer is usually 4 times the size of the Model Dimension (768 * 4). This gives the model more capacity to store "facts".
# ---------------------------------

class LiverQADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


# --- REPLACE THIS FUNCTION ---
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    # FIX: Ensure the scaling factor matches the query's data type (FP16 or FP32)
    scale = torch.sqrt(torch.tensor(d_k, device=query.device, dtype=query.dtype))

    scores = torch.matmul(query, key.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores + mask

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        query = split_heads(self.query(query))
        key = split_heads(self.key(key))
        value = split_heads(self.value(value))

        attention_output, _ = scaled_dot_product_attention(query, key, value, mask)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(attention_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, is_decoder_block=False):
        super().__init__()
        self.is_decoder_block = is_decoder_block

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        if self.is_decoder_block:
            self.cross_attention = MultiHeadAttention(d_model, num_heads)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        attn_output = self.dropout1(self.self_attention(x, x, x, self_attn_mask))
        x = self.norm1(x + attn_output)

        if self.is_decoder_block and encoder_output is not None:
            cross_attn_output = self.dropout2(self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask))
            x = self.norm2(x + cross_attn_output)

        ff_output = self.dropout3(self.feed_forward(x))
        x = self.norm3(x + ff_output)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)

# --- REPLACE THIS FUNCTION ---
def generate_square_subsequent_mask(sz, device, dtype=torch.float32):
    """Generates a causal mask (upper triangular -inf) with specific dtype."""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    # Convert to the correct dtype (FP16 or FP32) BEFORE filling
    mask = mask.to(dtype).masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class CustomTransformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder_block=False) for _ in range(num_layers)])

        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder_block=True) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # --- Encoder ---
        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)

        for layer in self.encoder_layers:
            src = layer(src, self_attn_mask=src_mask)

        encoder_output = src

        # --- Decoder ---
        trg = self.decoder_embedding(trg)
        trg = self.positional_encoding(trg)

        # Generate Causal Mask for Decoder
        trg_seq_len = trg.size(1)

        # --- FIX: Pass dtype=trg.dtype to match FP16/FP32 ---
        causal_mask = generate_square_subsequent_mask(trg_seq_len, trg.device, dtype=trg.dtype)

        if trg_mask is not None:
            causal_mask = causal_mask + trg_mask

        for layer in self.decoder_layers:
            trg = layer(trg, encoder_output=encoder_output, self_attn_mask=causal_mask, cross_attn_mask=src_mask)

        output = self.fc_out(trg)
        return output

# ... [Tokenizer and Data Loading functions remain unchanged] ...

def create_tokenizer(model_directory_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_directory_path)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit()

def load_gpt2_tokenizer():
    print("\nload_gpt2_tokenizer() - BEGIN")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        tokenizer_len = len(tokenizer)
        print(f"load_gpt2_tokenizer() - Vocab size: {tokenizer_len}")
        return tokenizer, tokenizer_len
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit()

# --- MODIFIED TRAINING FUNCTION FOR GPU ---
def model_training(epochs, train_dataloader, val_dataloader, device, optimizer, criterion, model, model_save_dir, patience=5):
    print("model_training() - BEGIN")
    model.to(device)

    # Optional: Compile if this function is called directly
    # model = torch.compile(model, mode="max-autotune")

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # --- RDNA 4 OPTIMIZATION: Scaler ---
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        model.train()
        train_total_loss = 0

        print(f"\nEpoch {epoch+1}/{epochs} [Training]", end="")

        for i, batch in enumerate(train_dataloader):
            src_data = batch['input_ids'].to(device)
            trg_data = batch['labels'].to(device)

            decoder_input = trg_data[:, :-1]
            labels = trg_data[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            min_len = min(decoder_input.size(1), labels.size(1))
            decoder_input = decoder_input[:, :min_len]
            labels = labels[:, :min_len]

            # --- RDNA 4 OPTIMIZATION: Autocast ---
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(src_data, decoder_input)
                output_flat = output.reshape(-1, output.shape[-1])
                labels_flat = labels.reshape(-1)
                loss = criterion(output_flat, labels_flat)

            # --- RDNA 4 OPTIMIZATION: Backward ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_total_loss += loss.item()

            if i % 10 == 0: print(".", end="", flush=True)

        train_avg_loss = train_total_loss / len(train_dataloader)
        print(f" Loss: {train_avg_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                src_data = batch['input_ids'].to(device)
                trg_data = batch['labels'].to(device)

                decoder_input = trg_data[:, :-1]
                labels = trg_data[:, 1:]

                min_len = min(decoder_input.size(1), labels.size(1))
                decoder_input = decoder_input[:, :min_len]
                labels = labels[:, :min_len]

                # Use autocast for validation too (faster)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(src_data, decoder_input)
                    loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

                val_total_loss += loss.item()

        val_avg_loss = val_total_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs} [Validation] Loss: {val_avg_loss:.4f}")

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            epochs_no_improve = 0
            os.makedirs(model_save_dir, exist_ok=True)
            best_model_path = os.path.join(model_save_dir, "model_best.safetensors")
            save_file(model.state_dict(), best_model_path)
            print(f"Saved best model. Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

# ... [The rest of the file (Data Loading, etc.) remains unchanged] ...
def read_csv_line_by_line(dataset_file_path):
    print(f"Loading CSV: {dataset_file_path}")
    if not os.path.exists(dataset_file_path):
        print("File not found.")
        return None
    try:
        df = pd.read_csv(dataset_file_path, encoding='utf-8', on_bad_lines='warn', engine='python')
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None


def load_general_knowledge_as_qa(path_pattern, num_samples=300, max_chars=1000):
    """
    Loads general text files and formats them strictly as Question/Answer pairs.
    INCLUDES AGGRESSIVE CLEANING FOR GUTENBERG EBOOKS.
    """
    print(f"Loading General Knowledge data from: {path_pattern}")
    files = glob.glob(path_pattern)

    if not files:
        print("No general knowledge files found. Skipping mixing.")
        return pd.DataFrame(columns=['question', 'answer'])

    selected_files = random.sample(files, min(num_samples, len(files)))
    print(f"Selected {len(selected_files)} general files for replay buffer.")

    questions = []
    answers = []

    for file_path in selected_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

                # --- 1. REMOVE GUTENBERG HEADERS/FOOTERS ---
                # Most Project Gutenberg files have these markers. We want the text BETWEEN them.
                if "*** START OF THIS PROJECT GUTENBERG" in text:
                    text = text.split("*** START OF THIS PROJECT GUTENBERG")[-1]
                if "*** START OF THE PROJECT GUTENBERG" in text:
                    text = text.split("*** START OF THE PROJECT GUTENBERG")[-1]

                if "*** END OF THIS PROJECT GUTENBERG" in text:
                    text = text.split("*** END OF THIS PROJECT GUTENBERG")[0]
                if "*** END OF THE PROJECT GUTENBERG" in text:
                    text = text.split("*** END OF THE PROJECT GUTENBERG")[0]

                # --- 2. CLEAN LINES ---
                lines = text.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    # Skip empty lines, short lines, or metadata lines
                    if len(line) < 50: continue
                    if "eBook" in line or "http" in line or "www." in line: continue
                    if "Project Gutenberg" in line: continue
                    if line.isupper(): continue  # Skip CHAPTER TITLES

                    clean_lines.append(line)

                text = " ".join(clean_lines)

                # Ensure we still have content
                if len(text) < 100:
                    continue

                # --- 3. CREATE QA PAIR ---
                text = text[:max_chars]
                split_idx = 100
                context_preview = text[:split_idx]
                rest_of_text = text[split_idx:]

                fake_question = f"Complete the following text: {context_preview}..."

                questions.append(fake_question)
                answers.append(rest_of_text)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return pd.DataFrame({'question': questions, 'answer': answers})


def save_model_weights(model, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    path = os.path.join(save_directory, "model.safetensors")
    save_file(model.state_dict(), path)
    print(f"Weights saved to {path}")

def create_model_configuration(model_save_dir, input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout):
    config = {
        "architectures": ["CustomTransformer"],
        "input_vocab_size": input_vocab_size,
        "target_vocab_size": target_vocab_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "max_len": max_len,
        "dropout": dropout
    }
    with open(os.path.join(model_save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("Config saved.")

def tokenize_dataset(tokenizer, questions, answers, max_length):
    bos_token = tokenizer.bos_token if tokenizer.bos_token else tokenizer.eos_token
    eos_token = tokenizer.eos_token

    processed_answers = [bos_token + " " + str(a) + " " + eos_token for a in answers]
    questions = [str(q) for q in questions]

    encoder_tokenized = tokenizer(questions, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
    decoder_tokenized = tokenizer(processed_answers, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")

    return {
        'input_ids': encoder_tokenized['input_ids'],
        'attention_mask': encoder_tokenized['attention_mask'],
        'labels': decoder_tokenized['input_ids']
    }

def get_dataloaders(liver_data_path, dict_data_path, general_data_path, tokenizer, batch_size, test_size=0.1, val_size=0.1):
    # 1. Load Liver Data (Domain Specific)
    liver_df = read_csv_line_by_line(liver_data_path)
    if liver_df is not None:
        liver_df = liver_df.dropna(subset=['question', 'answer'])
    else:
        liver_df = pd.DataFrame(columns=['question', 'answer'])

    # 2. Load Dictionary Data (Optional Domain Support)
    try:
        dict_df = pd.read_csv(dict_data_path)
        dict_df = dict_df.rename(columns={'word': 'question', 'definition': 'answer'})
        dict_df = dict_df.dropna(subset=['question', 'answer'])
    except Exception as e:
        print(f"Dictionary error or not found: {e}")
        dict_df = pd.DataFrame(columns=['question', 'answer'])

    # 3. Load General Knowledge Data (Replay Buffer for Catastrophic Forgetting)
    # We load approx 30% of general data relative to specific data size to keep a balance
    general_df = load_general_knowledge_as_qa(general_data_path, num_samples=300)

    # 4. Combine All
    df = pd.concat([liver_df, dict_df, general_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    print(f"--- Data Composition ---")
    print(f"Liver Data: {len(liver_df)}")
    print(f"Dictionary Data: {len(dict_df)}")
    print(f"General Knowledge (Replay): {len(general_df)}")
    print(f"Total Data Rows: {len(df)}")
    print(f"------------------------")

    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)

    def create_dl(split_df, shuffle):
        # Ensure data is string type before tokenization
        data = tokenize_dataset(tokenizer, split_df['question'].astype(str).tolist(), split_df['answer'].astype(str).tolist(), TOKENIZATION_MAX_LENGTH)
        ds = LiverQADataset(data)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True) 

    return create_dl(train, True), create_dl(val, False), create_dl(test, False)

def apply_repetition_penalty(logits, sequence, penalty=1.2):
    if sequence.size(0) != 1: return logits
    
    current_logits = logits[0]
    unique_tokens = sequence[0].unique()
    
    for token_id in unique_tokens:
        if current_logits[token_id] < 0:
            current_logits[token_id] *= penalty
        else:
            current_logits[token_id] /= penalty
    return logits

def generate_text_with_beam(model, tokenizer, prompt_text, max_output_length=50, beam_width=3, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    # Start with BOS/EOS
    start_token = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id
    current_seq = torch.full((1, 1), start_token, dtype=torch.long, device=device)
    
    beams = [(0.0, current_seq)] # (log_prob, sequence)
    finished_beams = []
    
    for step in range(max_output_length):
        candidates = []
        
        for log_prob, seq in beams:
            if seq[0, -1].item() == tokenizer.eos_token_id and seq.size(1) > 1:
                finished_beams.append((log_prob, seq))
                continue
                
            with torch.no_grad():
                output = model(input_ids, seq)
            
            next_logits = output[:, -1, :]
            next_logits = apply_repetition_penalty(next_logits, seq, 2.0)
            
            if temperature != 1.0:
                next_logits /= temperature
                
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            top_probs, top_ids = torch.topk(next_log_probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                new_log_prob = log_prob + top_probs[0, i].item()
                new_token = top_ids[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, new_token], dim=-1)
                candidates.append((new_log_prob, new_seq))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
        
        if all(seq[0, -1].item() == tokenizer.eos_token_id for _, seq in beams):
            break
            
    if finished_beams:
        best_seq = max(finished_beams, key=lambda x: x[0])[1]
    else:
        best_seq = beams[0][1]
        
    decoded = tokenizer.decode(best_seq.squeeze().tolist(), skip_special_tokens=True)
    return decoded