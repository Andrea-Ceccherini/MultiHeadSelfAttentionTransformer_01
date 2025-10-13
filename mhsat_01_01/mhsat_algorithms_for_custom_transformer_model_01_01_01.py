"""
MultiHeadAttention, FeedForward, TransformerBlock, PositionalEncoding, and CustomTransformer classes

this code implements the Multi-Head Attention algorithm, which is the core component of the Transformer architecture,
to create an LLM model based on supervised data for a Question and Answer (Q&A) task.


"""


import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F # For log_softmax
from transformers import AutoTokenizer
import torch.nn as nn
import os
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
import warnings
from torch.utils.data import DataLoader, Dataset


class LiverQADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])# Define the feedforward network


# ===========================Classes for ====================================
intermediate_gradients = [] # Global list to store gradients for debugging

# ------- 121,385,042 Parameters in total for the CustomTransformer Model - BEGIN ----
TOKENIZATION_MAX_LENGTH = 128
NUM_LAYERS = 6
D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
# -------- 121,385,042 Parameters in total for the CustomTransformer Model - END ----

# ------- 1,121,385,042 Parameters in total for the CustomTransformer Model - BEGIN ----
# TOKENIZATION_MAX_LENGTH = 128
# NUM_LAYERS = 15
# D_MODEL = 1024
# NUM_HEADS = 8
# D_FF = 4096
# DROPOUT = 0.1
# -------- 1,121,385,042 Parameters in total for the CustomTransformer Model - END ----

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
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

        # Self-attention layer
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention layer (only for decoder blocks)
        if self.is_decoder_block:
            self.cross_attention = MultiHeadAttention(d_model, num_heads)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        # Feed-forward layer
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)  # Third layer norm for decoder blocks
        self.dropout3 = nn.Dropout(dropout)  # Third dropout for decoder blocks

    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        # 1. Self-Attention Sub-layer
        # For encoder blocks: query, key, value all come from 'x'
        # For decoder blocks: query, key, value all come from 'x' (masked self-attention)
        attn_output = self.dropout1(self.self_attention(x, x, x, self_attn_mask))
        x = self.norm1(x + attn_output)

        # 2. Cross-Attention Sub-layer (only for decoder blocks)
        if self.is_decoder_block and encoder_output is not None:
            # Query from decoder's current output (x)
            # Key and Value from encoder's final output (encoder_output)
            cross_attn_output = self.dropout2(self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask))
            x = self.norm2(x + cross_attn_output)

        # 3. Feed-Forward Sub-layer
        ff_output = self.dropout3(self.feed_forward(x))
        x = self.norm3(x + ff_output)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).requires_grad_(False)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_to_add = self.pe[:, :x.size(1)].to(x.device)
        return x + pe_to_add



class CustomTransformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=100,
                 dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Encoder layers: These are not decoder blocks, so is_decoder_block=False
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder_block=False) for _ in range(num_layers)])

        # Decoder layers: These ARE decoder blocks, so is_decoder_block=True
        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder_block=True) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # Encoder forward pass
        src = self.encoder_embedding(src)
        if src.requires_grad:
            src.retain_grad()  # Add this line
            src.register_hook(save_gradient_hook('encoder_embedding_output'))

        src = self.positional_encoding(src)
        if src.requires_grad:
            src.retain_grad()  # Add this line
            src.register_hook(save_gradient_hook('encoder_pos_encoding_output'))

        for i, layer in enumerate(self.encoder_layers):
            src = layer(src, self_attn_mask=src_mask)  # Encoder blocks only need self_attn_mask
            # Add debugging hook if needed here for encoder_layer_i_output

        # The output of the encoder becomes the 'encoder_output' for the decoder
        encoder_output = src

        # Decoder forward pass
        trg = self.decoder_embedding(trg)
        # Add debugging hook if needed here
        trg = self.positional_encoding(trg)
        # Add debugging hook if needed here

        for i, layer in enumerate(self.decoder_layers):
            # Decoder blocks need their own self-attention mask (trg_mask) AND
            # the encoder_output for cross-attention.
            # No specific cross_attn_mask needed for now, as it usually derived from src_mask
            trg = layer(trg, encoder_output=encoder_output, self_attn_mask=trg_mask, cross_attn_mask=src_mask)
            # Add debugging hook if needed here for decoder_layer_i_output

        # Final output projection
        output = self.fc_out(trg)
        # Add debugging hook if needed here for final_output_logits

        return output


def save_gradient_hook(name):
    """
    Creates a hook function to save the sum of gradients of a tensor.
    """
    def hook(grad):
        if grad is not None:
            intermediate_gradients.append((name, grad.sum().item()))
        else:
            intermediate_gradients.append((name, "None"))
    return hook


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32).to(query.device))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights



def create_tokenizer(model_directory_path_):
    """
    Creates and returns a tokenizer from the specified model directory.
    """
    print(f"\ncreate_tokenizer() - BEGIN")
    print(f"create_tokenizer() - model_directory_path_ = {model_directory_path_}")
    try:
        tokenizer_ = AutoTokenizer.from_pretrained(model_directory_path_)
        print("create_tokenizer() - END\n")
        return tokenizer_
    except Exception as e:
        print(f"create_tokenizer() - Error loading tokenizer: {e}")
        exit()

# ============================ Functions for Creation Begin ===================================

def model_training(epochs, train_dataloader, val_dataloader, device, optimizer, criterion, model, model_save_dir,
                   patience=5):
    """
    Trains the CustomTransformer model for a specified number of epochs with validation and early stopping.

    Args:
        epochs (int): Number of training epochs.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): Device to train on ('cuda' or 'cpu').
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Loss function.
        model (torch.nn.Module): The custom Transformer model.
        model_save_dir (str): Directory to save the best model weights.
        patience (int): Number of epochs to wait for validation loss improvement before early stopping.
    """
    print("model_training() - BEGIN")
    print("model_training() - Setting model to training mode - STARTED")
    model.train()  # Set the model to training mode
    print("model_training() - Setting model to training mode - COMPLETED")

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Check requires_grad for a parameter (optional, for initial debugging confirmation)
    # print("model_training() - Linear1 weight requires_grad:", model.encoder_layers[0].feed_forward.linear1.weight.requires_grad)

    for epoch in range(epochs):
        # ---------------------
        #   Training Phase
        # ---------------------
        model.train()  # Ensure model is in training mode
        train_total_loss = 0
        for batch in train_dataloader:
            # global intermediate_gradients # Uncomment if still using hooks
            # intermediate_gradients = [] # Uncomment if still using hooks

            src_data = batch['input_ids'].to(device)
            trg_data = batch['labels'].to(device)

            optimizer.zero_grad()

            decoder_input = trg_data[:, :-1]
            labels_for_loss = trg_data[:, 1:]

            output = model(src_data, decoder_input)

            output_flat = output.reshape(-1, output.shape[-1])
            labels_flat = labels_for_loss.reshape(-1)

            loss = criterion(output_flat, labels_flat)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()

        train_avg_loss = train_total_loss / len(train_dataloader)
        print(f"model_training() - Epoch {epoch + 1}/{epochs}, Train Average Loss: {train_avg_loss:.4f}")

        # ---------------------
        #   Validation Phase
        # ---------------------
        model.eval()  # Set model to evaluation mode
        val_total_loss = 0
        with torch.no_grad():  # Disable gradient calculations during validation
            for batch in val_dataloader:
                src_data = batch['input_ids'].to(device)
                trg_data = batch['labels'].to(device)

                decoder_input = trg_data[:, :-1]
                labels_for_loss = trg_data[:, 1:]

                output = model(src_data, decoder_input)

                output_flat = output.reshape(-1, output.shape[-1])
                labels_flat = labels_for_loss.reshape(-1)

                loss = criterion(output_flat, labels_flat)
                val_total_loss += loss.item()

        val_avg_loss = val_total_loss / len(val_dataloader)
        print(f"model_training() - Epoch {epoch + 1}/{epochs}, Validation Average Loss: {val_avg_loss:.4f}")

        # ---------------------
        #   Early Stopping Logic
        # ---------------------
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            epochs_no_improve = 0
            # Save the best model
            # Ensure the directory exists before saving
            os.makedirs(model_save_dir, exist_ok=True)
            best_model_path = os.path.join(model_save_dir, "model_best.safetensors")
            save_file(model.state_dict(), best_model_path)
            print(f"model_training() - Saved best model to {best_model_path} with Validation Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"model_training() - Validation loss did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"model_training() - Early stopping triggered after {patience} epochs without improvement.")
                break  # Exit the training loop

    print("model_training() - END")

def read_csv_line_by_line(dataset_file_path_):
    """
    Reads a CSV file line by line, loading it into a pandas DataFrame.
    It skips problematic lines and prints warnings for them.

    Args:
        dataset_file_path_ (str): The path to the CSV file.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame if successful, None otherwise.
    """
    print("\nread_csv_line_by_line() - BEGIN")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_colwidth', None)
    if not os.path.exists(dataset_file_path_):
        print(f"Error: The file '{dataset_file_path_}' was not found.")
        print("Please ensure the CSV file exists at the specified path.")
        return None  # Return None if file is not found

    df_ = None  # Initialize df_
    try:
        # Use warnings.catch_warnings to capture warnings
        with warnings.catch_warnings(record=True) as w:
            # Ensure all warnings are caught
            warnings.simplefilter("always")

            # Attempt to read the CSV, explicitly specifying UTF-8 encoding.
            # on_bad_lines='warn' will trigger a warning for each problematic line encountered
            # engine='python' is required for on_bad_lines to function properly.
            print(f"read_csv_line_by_line() - Attempting to load CSV file: '{dataset_file_path_}'")
            df_ = pd.read_csv(dataset_file_path_, encoding='utf-8', on_bad_lines='warn', engine='python')

            # Print any captured warnings as normal messages
            if w:
                print("\nread_csv_line_by_line() - Warnings encountered during CSV loading ---")
                for warning_message in w:
                    # Filter for pandas ParserWarnings specifically if desired,
                    # or print all captured warnings
                    if issubclass(warning_message.category, pd.errors.ParserWarning):
                        print(f"read_csv_line_by_line() - Bad line detected (ParserWarning): {warning_message.message}")
                    else:
                        print(f"read_csv_line_by_line() - Warning: {warning_message.message}")
                print("----------------------------------------------")

            print("\nread_csv_line_by_line() - CSV file loaded successfully!")
            print(f"read_csv_line_by_line() - DataFrame shape (rows, columns): {df_.shape}")
            print("\nread_csv_line_by_line() - First 5 rows of the DataFrame:")
            print(df_.head())

    except UnicodeDecodeError as e:
        print(f"\nread_csv_line_by_line() - UnicodeDecodeError occurred: {e}")
        print("read_csv_line_by_line() - This typically means the file's encoding is not UTF-8.")
        print("read_csv_line_by_line() - Try a different encoding like 'latin1' or 'cp1252' if UTF-8 doesn't work for your file.")
    except pd.errors.ParserError as e:
        print(f"\nread_csv_line_by_line() - A Pandas parsing error occurred: {e}")
        print("read_csv_line_by_line() - This indicates inconsistent CSV formatting that even 'on_bad_lines' couldn't fully handle.")
        print("read_csv_line_by_line() - You might need to manually inspect the file for severe formatting issues.")
    except Exception as e:
        print(f"\nread_csv_line_by_line() - An unexpected error occurred: {e}")

    print("\nread_csv_line_by_line() - END")

    return df_

def save_model_weights(model, save_directory):
    print("\nsave_model_weights() - BEGIN")
    safetensors_file_path = os.path.join(save_directory, "model.safetensors")
    print("save_model_weights() - safetensors_file_path =", safetensors_file_path)

    save_file(model.state_dict(), safetensors_file_path)
    print(f"Model weights saved to {safetensors_file_path} using Safetensors!")
    print("save_model_weights() - END\n")


def create_model_configuration(model_save_dir_, input_vocab_size_, target_vocab_size_, d_model_, num_heads_, d_ff_,
                               num_layers_, max_len_, dropout_):
    print("create_model_configuration() - BEGIN")
    automodel_ = "custom_transformer_algorithm.CustomTransformer"
    model_config_ = {
        "architectures": ["CustomTransformer"],
        "auto_map": {
            "AutoModel": automodel_,
        },
        "input_vocab_size": input_vocab_size_,
        "target_vocab_size": target_vocab_size_,
        "d_model": d_model_,
        "num_heads": num_heads_,
        "d_ff": d_ff_,
        "num_layers": num_layers_,
        "max_len": max_len_,
        "dropout": dropout_
    }

    config_path_ = os.path.join(model_save_dir_, "config.json")
    with open(config_path_, "w") as f:
        json.dump(model_config_, f, indent=4)
    print(f"create_model_configuration() - Model configuration saved to {config_path_}.")

    print("create_model_configuration() - END", )


def tokenize_dataset(tokenizer_, questions_, answers_, max_length_):
    print("tokenize_dataset() - BEGIN")
    tokenized_data_ = tokenizer_(
        questions_,
        text_target=answers_,
        truncation=True,
        padding='max_length',
        max_length=max_length_
    )
    print("tokenize_dataset() - END")
    return tokenized_data_


def get_dataloaders(dataset_file_path_, tokenizer_, batch_size_, test_size=0.1, val_size=0.1):
    """
    Creates DataLoaders for training, validation, and test sets.
    """
    print("get_dataloaders() - BEGIN")
    df = read_csv_line_by_line(dataset_file_path_)
    df = df.dropna(subset=['question', 'answer'])
    print(f"get_dataloaders() - Initial DataFrame shape: {df.shape}")

    # Split into train + validation and test sets
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Split train + validation into train and validation sets
    # Adjust val_size relative to the remaining data
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, random_state=42)

    print(f"get_dataloaders() - Train shape: {train_df.shape}")
    print(f"get_dataloaders() - Validation shape: {val_df.shape}")
    print(f"get_dataloaders() - Test shape: {test_df.shape}")

    # Process each split
    train_questions, train_answers = train_df['question'].tolist(), train_df['answer'].tolist()
    val_questions, val_answers = val_df['question'].tolist(), val_df['answer'].tolist()
    test_questions, test_answers = test_df['question'].tolist(), test_df['answer'].tolist()

    # Tokenize datasets
    train_tokenized_data = tokenize_dataset(tokenizer_, train_questions, train_answers, max_length_=TOKENIZATION_MAX_LENGTH)
    val_tokenized_data = tokenize_dataset(tokenizer_, val_questions, val_answers, max_length_=TOKENIZATION_MAX_LENGTH)
    test_tokenized_data = tokenize_dataset(tokenizer_, test_questions, test_answers,max_length_=TOKENIZATION_MAX_LENGTH)  # Optional: for later evaluation

    # Create Dataset objects
    train_dataset = LiverQADataset(train_tokenized_data)
    val_dataset = LiverQADataset(val_tokenized_data)
    test_dataset = LiverQADataset(test_tokenized_data)  # Optional

    # Create DataLoaders (no DataCollatorForSeq2Seq for custom model)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_, shuffle=False)  # No need to shuffle validation
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_, shuffle=False)  # No need to shuffle test

    print("get_dataloaders() - END")

    return train_dataloader, val_dataloader, test_dataloader




def apply_repetition_penalty(next_token_logits, sequence, penalty_factor=1.2):
    """
    Applica una penalità ai logit dei token che sono già presenti nella sequenza.

    Args:
        next_token_logits (torch.Tensor): Logit per il token successivo, di forma (batch_size, vocab_size).
        sequence (torch.Tensor): La sequenza di token generata finora, di forma (batch_size, sequence_length).
        penalty_factor (float): Il fattore di penalità. Un valore > 1 penalizza la ripetizione.

    Returns:
        torch.Tensor: Logit modificati.
    """
    print("apply_repetition_penalty() - BEGIN")


    # 1. Estrai gli ID dei token da penalizzare (esclusi gli speciali come BOS)
    # Assumiamo batch_size=1 (tipico nell'inferenza interattiva)
    if sequence.dim() == 2 and sequence.size(0) == 1:
        # sequence.squeeze(0) prende la sequenza di token.
        # .unique() assicura che penalizziamo un token una sola volta, anche se ripetuto.
        repeated_token_ids = sequence.squeeze(0).unique()
    else:
        # Gestione di batch_size > 1 o formato inatteso (semplificazione per questo script)
        # Qui potresti implementare una logica più complessa per il batching.
        return next_token_logits

    # 2. Applica la penalità
    # Creiamo un tensore di zeri delle dimensioni del vocabulario

    # next_token_logits è un tensore 1D (per batch_size=1) o 2D (batch_size, vocab_size).
    # Se il tensore è 2D, applichiamo la penalità al primo (e unico) elemento del batch.

    # 2.1 Prendi i logit della prima (e unica) riga del batch
    current_logits = next_token_logits[0]

    # 2.2 Itera sugli ID dei token ripetuti e modifica i logit
    for token_id in repeated_token_ids:
        token_id = token_id.item()  # Estrae il valore intero
        logit = current_logits[token_id]

        if logit < 0:
            # Penalizza i token poco probabili moltiplicandoli (per rendere il logit più negativo)
            current_logits[token_id] = logit * penalty_factor
        else:
            # Penalizza i token probabili dividendoli (per renderli meno probabili)
            current_logits[token_id] = logit / penalty_factor

    # La funzione restituisce il tensore modificato
    print(f"apply_repetition_penalty() - current_logits: {current_logits}")
    print("apply_repetition_penalty() - END")
    return next_token_logits




def generate_text_with_beam(model_, tokenizer_, prompt_text_, max_output_length_=50, beam_width_=3, temperature_=1.0):
    """
    Generates text using the trained CustomTransformer model with beam search and temperature sampling.

    Args:
        model_ (nn.Module): The trained CustomTransformer model.
        tokenizer_ (AutoTokenizer): The tokenizer used with the model.
        prompt_text_ (str): The input text prompt.
        max_output_length_ (int): The maximum number of tokens to generate.
        beam_width_ (int): The number of top sequences to keep at each step.
        temperature_ (float): Controls the randomness of predictions.
                              Higher values (e.g., 1.5) make output more random.
                              Lower values (e.g., 0.7) make it more deterministic.
                              Value of 1.0 means no change to logits.

    Returns:
        str: The generated text.
    """
    print("\ngenerate_text() - BEGIN ")
    print("generate_text() - model_ =", model_)
    print("generate_text() - prompt_text_ =", prompt_text_)
    print("generate_text() - max_output_length_ =", max_output_length_)
    print("generate_text() - beam_width_ =", beam_width_)
    print("generate_text() - temperature_ =", temperature_)

    model_.eval()  # Set model to evaluation mode
    device = next(model_.parameters()).device  # Get the device the model is on
    print("generate_text() - device =", device)


    with torch.no_grad():  # Disable gradient calculations during inference

        # Encode the input prompt for the encoder of the transformer
            # Takes your raw text prompt (e.g., "Where is placed the Liver").
            # Tokenizes it (splits it into subwords/words and converts them into numerical IDs).
            # Adds any necessary special tokens (like [CLS] or [SEP]).
            # Pads or truncates the sequence to a fixed length (if configured).
            # Wraps these numerical IDs into a PyTorch tensor.
            # Moves that tensor from the CPU to the GPU (if device is set to cuda) so that your Transformer model can
                # process it efficiently.
        return_tensors_ = "pt"  # PyTorch tensors
        input_ids = tokenizer_.encode(prompt_text_, return_tensors=return_tensors_).to(device)
        print("generate_text() - input_ids =", input_ids)


        # Initialize beams: Each beam is (log_probability, sequence_tensor)
        # Start with the EOS token (which acts as BOS for GPT-2 tokenizer)
        # Retrieve the numerical ID of the "End of Sentence" (EOS) token from tokenizers vocabulary.
        initial_decoder_input_id = tokenizer_.eos_token_id  # Typically 50256 for GPT-2
        print("generate_text() - initial_decoder_input_id =", initial_decoder_input_id)

        # creates the initial input tensor for the Transformer's decoder during inference.
        initial_sequence = torch.full((1, 1), initial_decoder_input_id, dtype=torch.long).to(device)
        print("generate_text() - initial_sequence (tensor) =", initial_sequence)

        # Each beam is a tuple of (log_probability, sequence_tensor)
        # initializing the beam search algorithm with a single beam that contains the initial token.
        beams = [(0.0, initial_sequence)]  # (log_probability, sequence)
        print("generate_text() - beams =", beams)

        finished_beams = []  # Store completed sequences that hit EOS

        print("\ngenerate_text() - --------------------- manual check BEGIN-----------------------------\n")
        print("\ngenerate_text() - DEBUG: First Generation Step (Manual Check) ---")
        print(f"generate_text() - Initial decoder input (token ID) = {initial_decoder_input_id}")

        a_ = initial_sequence.squeeze().tolist()
        b_ = tokenizer_.decode(a_, skip_special_tokens=False)
        print(f"generate_text() - Initial decoder input (decoded, with special tokens) = '{b_}'")

        # forward pass through Transformer model to get its raw predictions (logits) for the next token.
        simulated_output_logits = model_(input_ids, initial_sequence)
        print("generate_text() - Simulated output logits =", simulated_output_logits)

        simulated_next_token_logits = simulated_output_logits[:, -1, :]
        print("generate_text() - Simulated next token logits =", simulated_next_token_logits)

        # Apply temperature for debugging check as well
        if temperature_ != 1.0:
            simulated_next_token_logits = simulated_next_token_logits / temperature_
        print("generate_text() - Simulated next token logits =", simulated_next_token_logits)

        # Apply log_softmax to get log probabilities
        simulated_next_token_log_probs = F.log_softmax(simulated_next_token_logits, dim=-1)
        print("generate_text() - Simulated next token log probabilities =", simulated_next_token_log_probs)

        # Get the top 'beam_width_' next tokens and their log probabilities
        simulated_top_log_probs, simulated_top_token_ids = torch.topk(simulated_next_token_log_probs, beam_width_, dim=-1)
        print("generate_text() - Simulated Top Log Probabilities =", simulated_top_log_probs)

        print(f"generate_text() - Simulated Top {beam_width_} predicted next token IDs: {simulated_top_token_ids.tolist()}")

        c_ = [tokenizer_.decode([id_], skip_special_tokens=False) for id_ in simulated_top_token_ids.squeeze().tolist()]
        print(f"generate_text() - Simulated Top {beam_width_} predicted next tokens (decoded, with special tokens): {c_}")
        print("generate_text() - --------------------- manual check END-----------------------------\n")

        print("generate_text() - Main beam search loop BEGIN")
        numbers_of_steps_ = int(max_output_length_) # Cast to int here
        print("generate_text() - numbers_of_steps_ =", numbers_of_steps_)
        for step in range(numbers_of_steps_):
            print(f"\ngenerate_text() - Step {step + 1}/{numbers_of_steps_} - BEGIN")
            all_candidates_ = []  # Store all possible next beam candidates for this step

            for log_prob, sequence in beams:
                print(f"generate_text() - Current beam sequence: {sequence}, Log Probability: {log_prob} - BEGIN")
                # Only mark as "finished" and stop extending if EOS (End Of Sequence) is predicted *after* the initial token
                if sequence[0, -1].item() == tokenizer_.eos_token_id and sequence.size(1) > 1:
                    finished_beams.append((log_prob, sequence))
                    continue  # Do not extend this beam further

                print("generate_text() - Pass encoder input and current decoder sequence to the model - BEGIN")
                output_logits_ = model_(input_ids, sequence)
                print("generate_text() - Pass encoder input and current decoder sequence to the model - END")

                # Get the logits for the *last* token in the generated sequence,
                # which represents the prediction for the next token.
                print("generate_text() - extract the raw, un-normalized prediction scores logits) for the next token - BEGIN")
                next_token_logits = output_logits_[:, -1, :] # Shape: (1, vocab_size)
                print("generate_text() - extract the raw, un-normalized prediction scores logits) for the next token - END")


                REPETITION_PENALTY = 1.2

                # Applica la funzione ai logit prima della temperatura
                next_token_logits = apply_repetition_penalty(
                    next_token_logits,
                    sequence,
                    REPETITION_PENALTY
                )



                # --- Apply Temperature to Logits ---
                if temperature_ != 1.0:
                    next_token_logits = next_token_logits / temperature_
                # --- End Temperature Application ---

                # Apply log_softmax to get log probabilities
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

                # Get the top 'beam_width_' next tokens and their log probabilities
                top_log_probs, top_token_ids = torch.topk(next_token_log_probs, beam_width_, dim=-1)

                print("generate_text() - Expand each current beam with the top 'beam_width_' next tokens - BEGIN")
                for i in range(beam_width_):
                    new_log_prob = log_prob + top_log_probs[0, i].item()
                    new_token_id = top_token_ids[0, i].unsqueeze(0).unsqueeze(0)  # Make it 1x1 tensor
                    new_sequence = torch.cat([sequence, new_token_id], dim=-1)
                    all_candidates_.append((new_log_prob, new_sequence))
                print("generate_text() - Expand each current beam with the top 'beam_width_' next tokens - END")
                print(f"generate_text() - Current beam sequence: {sequence}, Log Probability: {log_prob} - END")
            # Sort all candidates by their log probability (higher is better)
            all_candidates_.sort(key=lambda x: x[0], reverse=True)

            # Select the top 'beam_width_' candidates for the next iteration
            beams = all_candidates_[:beam_width_]
            print(f"generate_text() - After Step {step + 1}, Top {beam_width_} Beams Selected, {beams}.")

            # Debug: Print top beam's sequence at each step
            print(f"generate_text() - DEBUG: Step {step + 1} Beams - BEGIN")
            for i, (log_p, seq) in enumerate(beams):
                decoded_seq = tokenizer_.decode(seq.squeeze().tolist(), skip_special_tokens=False)
                print(f"generate_text() - Beam {i + 1} (LogProb: {log_p:.4f}): '{decoded_seq}'")
            print(f"generate_text() - DEBUG: Step {step + 1} Beams - END")

            # Termination condition for the loop: if all active beams have finished (all ended with EOS)
            # or if we've reached max length.
            if all(seq[0, -1].item() == tokenizer_.eos_token_id and seq.size(1) > 1 for _, seq in
                   beams) or step == max_output_length_ - 1:
                print("\ngenerate_text() - All beams have finished or max length reached. Terminating beam search.")
                print(f"generate_text() - Step {step + 1}/{numbers_of_steps_} - END\n")
                break  # Exit the loop, then process finished_beams/beams
            print(f"generate_text() - Step {step + 1}/{numbers_of_steps_} - END\n")
        print("generate_text() - Main beam search loop END\n")

        # After the loop, pick the best sequence from finished_beams or active beams
        if finished_beams:
            print("generate_text() - Sort finished beams by probability and pick the best one - BEGIN")
            finished_beams.sort(key=lambda x: x[0], reverse=True)
            best_sequence_ids = finished_beams[0][1]
            print("generate_text() - Sort finished beams by probability and pick the best one - END")
        else:
            # If no beam finished with EOS, take the best of the current (incomplete) beams
            print("generate_text() - No finished beams. Picking the best from current beams - BEGIN")
            best_sequence_ids = beams[0][1]
            print("generate_text() - No finished beams. Picking the best from current beams - END")

        # --- Final Decoding Step ---
        # Ensure decoded_tokens_list is always a list, even for single-element tensors.
        decoded_tokens_list = best_sequence_ids.flatten().tolist() # Convert tensor to a flat list of token IDs
        print(f"generate_text() - decoded_tokens_list = {decoded_tokens_list}")

        # Print the raw decoded sequence (with special tokens) for debugging
        raw_decoded_text = tokenizer_.decode(decoded_tokens_list, skip_special_tokens=False)
        print(f"\ngenerate_text() - raw_decoded_text = '{raw_decoded_text}'")

        # Strip the initial BOS token (if it was EOS)
        if decoded_tokens_list and decoded_tokens_list[0] == initial_decoder_input_id:
            print("generate_text() - Stripping initial BOS (Beginning of Sentence) token from the output. - BEGIN")
            decoded_tokens_list = decoded_tokens_list[1:]
            print("generate_text() - Stripping initial BOS (Beginning of Sentence) token from the output. - END")

        # Decode the rest, skipping other special tokens (like PAD if present)
        generated_text_ = tokenizer_.decode(decoded_tokens_list, skip_special_tokens=True)

        print("generate_text() - END\n")
        return generated_text_


def load_gpt2_tokenizer():
    print("\nload_gpt2_tokenizer() - BEGIN")
    try:
        tokenizer_ = AutoTokenizer.from_pretrained("gpt2")

        if tokenizer_.pad_token is None:
            tokenizer_.add_special_tokens({'pad_token': '[PAD]'})

        tokenizer_len_ = len(tokenizer_)
        print(f"load_gpt2_tokenizer() - Loaded GPT-2 tokenizer with vocabulary size: {tokenizer_len_}")
        print("load_gpt2_tokenizer() - END\n")
        return tokenizer_, tokenizer_len_
    except Exception as e:
        print(f"load_gpt2_tokenizer() - Error loading tokenizer: {e}")
        exit()