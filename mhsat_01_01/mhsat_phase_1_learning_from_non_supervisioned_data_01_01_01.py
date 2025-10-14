import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import os
from safetensors.torch import save_file
from mhsat_01_01.mhsat_algorithms_for_custom_transformer_model_01_01_01 import CustomTransformer, \
    NUM_LAYERS, TOKENIZATION_MAX_LENGTH, D_MODEL, NUM_HEADS, D_FF, DROPOUT, load_gpt2_tokenizer


# A simple Dataset class for unsupervised text
class UnsupervisedTextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


# Assume all your classes (CustomTransformer, etc.) are defined above this part.
# ... (your original classes here) ...


def model_training_unsupervised(epochs, dataloader, device, optimizer, criterion, model, model_save_dir, patience=5):
    """
    Trains the CustomTransformer model in an unsupervised manner (next-token prediction).

    Args:
        epochs (int): Number of training epochs.
        dataloader (torch.utils.data.DataLoader): DataLoader for unsupervised text data.
        device (torch.device): Device to train on ('cuda' or 'cpu').
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Loss function.
        model (torch.nn.Module): The custom Transformer model.
        model_save_dir (str): Directory to save the best model weights.
        patience (int): Number of epochs to wait for validation loss improvement before early stopping.
    """
    print("model_training_unsupervised() - BEGIN")
    model.train()  # Set the model to training mode

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # In un ambiente di pre-addestramento reale, avresti anche un dataloader di validazione
    # per semplicità usiamo lo stesso, ma in produzione sarebbe un set separato.

    for epoch in range(epochs):
        train_total_loss = 0

        # ---------------------
        #   Training Phase
        # ---------------------
        for batch in dataloader:
            # Qui il 'batch' contiene solo un'unica sequenza di testo
            # Esempio: "Il gatto sta..."
            src_data = batch['input_ids'].to(device)

            optimizer.zero_grad()

            # Decoder input: è la stessa sequenza ma senza l'ultimo token
            # Esempio: "Il gatto sta"
            decoder_input = src_data[:, :-1]

            # Label per il calcolo della loss: è la stessa sequenza ma senza il primo token
            # Esempio: " gatto sta..."
            labels_for_loss = src_data[:, 1:]

            # Forward pass
            output = model(src_data, decoder_input)

            # Calcolo della loss
            output_flat = output.reshape(-1, output.shape[-1])
            labels_flat = labels_for_loss.reshape(-1)
            loss = criterion(output_flat, labels_flat)

            # Backpropagation e ottimizzazione
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()

        train_avg_loss = train_total_loss / len(dataloader)
        print(f"model_training_unsupervised() - Epoch {epoch + 1}/{epochs}, Train Average Loss: {train_avg_loss:.4f}")

        # ---------------------
        #   Validation Phase
        # ---------------------
        model.eval()  # Set model to evaluation mode
        val_total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                src_data = batch['input_ids'].to(device)

                decoder_input = src_data[:, :-1]
                labels_for_loss = src_data[:, 1:]

                output = model(src_data, decoder_input)

                output_flat = output.reshape(-1, output.shape[-1])
                labels_flat = labels_for_loss.reshape(-1)

                loss = criterion(output_flat, labels_flat)
                val_total_loss += loss.item()

        val_avg_loss = val_total_loss / len(dataloader)
        print(
            f"model_training_unsupervised() - Epoch {epoch + 1}/{epochs}, Validation Average Loss: {val_avg_loss:.4f}")

        # Early stopping logic here (same as your original code)
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            epochs_no_improve = 0
            os.makedirs(model_save_dir, exist_ok=True)
            best_model_path = os.path.join(model_save_dir, "unsupervised_model_best.safetensors")
            save_file(model.state_dict(), best_model_path)
            print(f"model_training_unsupervised() - Saved best model with Validation Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"model_training_unsupervised() - Early stopping triggered.")
                break

    print("model_training_unsupervised() - END")


if __name__ == "__main__":
    print("__main__() - BEGIN")
    # Sostituisci "gpt2" con il tokenizer che intendi usare

    tokenizer, tokenizer_len = load_gpt2_tokenizer()

    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #
    # # === FIX: ADD THE SPECIAL TOKEN HERE, BEFORE MODEL INSTANTIATION ===
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # ===================================================================

    # Parametri del modello
    d_model = D_MODEL
    num_heads = NUM_HEADS
    d_ff = D_FF
    num_layers = NUM_LAYERS
    max_len = TOKENIZATION_MAX_LENGTH
    dropout = DROPOUT
    input_vocab_size = tokenizer_len
    target_vocab_size = tokenizer_len

    print(f"__main__() - d_model: {d_model}")
    print(f"__main__() - num_heads: {num_heads}")
    print(f"__main__() - d_ff: {d_ff}")
    print(f"__main__() - num_layers: {num_layers}")
    print(f"__main__() - max_len: {max_len}")
    print(f"__main__() - dropout: {dropout}")
    print(f"__main__() - input_vocab_size: {input_vocab_size}")
    print(f"__main__() - target_vocab_size: {target_vocab_size}")


    # Istanzia il modello
    model = CustomTransformer(
        input_vocab_size=input_vocab_size,
        target_vocab_size=input_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout
    )

    # --- NEW: Calculate and print the number of model parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n__main__() - Model contains {total_params:,} parameters.")
    # -------------------------------------------------------------

    # Sposta il modello sulla GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"__main__() - device: {device}")

    model.to(device)

    # --- Prepare your raw text data from a folder ---
    text_data_folder = "../../../Datasets/Books/"
    if not os.path.exists(text_data_folder):
        print(f"__main__() - Error: The folder '{text_data_folder}' was not found.")
        exit()

    raw_text_data = []
    for filename in os.listdir(text_data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_data_folder, filename)

            # 🎯 Modifica qui: Usa la codifica 'latin-1' (o 'windows-1252')
            with open(file_path, 'r', encoding='latin-1') as f:
                raw_text_data.append(f.read())

    if not raw_text_data:
        print(f"__main__() - Error: No text files found in '{text_data_folder}'.")
        exit()

    # Calculate and print the total number of words
    total_words = 0
    for text in raw_text_data:
        # Split the text by whitespace and add the number of words
        total_words += len(text.split())

    print(f"__main__() - Number of documents loaded: {len(raw_text_data)}")
    print(f"__main__() - Total number of words loaded: {total_words}")
    # -------------------------------------------------------------

    # 2. Tokenize the data
    tokenized_data = tokenizer(
        raw_text_data,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_len
    )

    # 3. Create a Dataset and a DataLoader
    unsupervised_dataset = UnsupervisedTextDataset(tokenized_data)
    unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=2, shuffle=True)

    # 4. Define the optimizer and loss function
    # L'ottimizzatore deve essere definito DOPO il ridimensionamento dei parametri del modello
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # The CrossEntropyLoss 'ignore_index' is crucial to ignore padding tokens
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # 5. Define training parameters
    epochs = 5
    model_save_dir = "unsupervised_model_weights/unsupervised_model_weights"
    patience = 3

    # 6. Call the training function to start the unsupervised training
    print("\n__main__() - Unsupervised training process - BEGIN\n")
    model_training_unsupervised(
        epochs=epochs,
        dataloader=unsupervised_dataloader,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        model=model,
        model_save_dir=model_save_dir,
        patience=patience
    )
    print("\n__main__() - Unsupervised training process - END\n")

    print("__main__() - END")
