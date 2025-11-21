import os
import json
from safetensors.torch import load_file
from transformers import AutoTokenizer
import torch

from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    generate_text_with_beam, CustomTransformer
)

def create_tokenizer_safe(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
    return tok

if __name__ == "__main__":
    print("INFERENCE - BEGIN")

    model_dir = "supervised_qa_model_files"
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found.")
        exit()

    # 1. Load Config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    # 2. Load Tokenizer
    tokenizer = create_tokenizer_safe(model_dir)

    # 3. Instantiate Model
    model = CustomTransformer(
        input_vocab_size=config["input_vocab_size"],
        target_vocab_size=config["target_vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        max_len=config["max_len"],
        dropout=config.get("dropout", 0.1)
    )

    # 4. Load Weights
    weights_path = os.path.join(model_dir, "model_best.safetensors")
    if not os.path.exists(weights_path):
        # Fallback to model.safetensors
        weights_path = os.path.join(model_dir, "model.safetensors")
    
    print(f"Loading weights from {weights_path}")
    model.load_state_dict(load_file(weights_path))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Model ready.")

    # 5. Inference Loop
    while True:
        user_input = input("\nQuestion (type 'exit' to quit): ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # --- MODIFICA FONDAMENTALE: FORMATTAZIONE DEL PROMPT ---
        # Il modello è stato addestrato su "Question: ... \nAnswer: ..."
        # Dobbiamo imitare questo formato per attivare la risposta corretta.
        formatted_prompt = f"Question: {user_input}\nAnswer:"
        
        # Parametri suggeriti:
        # temperature=0.7 -> Buon bilanciamento tra creatività e precisione
        # beam_width=3 o 5 -> Più alto è, più accurato (ma lento)
        full_response = generate_text_with_beam(
            model, tokenizer, formatted_prompt, 
            max_output_length=100, beam_width=5, temperature=0.7
        )
        
        # --- PULIZIA DELL'OUTPUT ---
        # La funzione restituisce tutto il testo (Prompt + Risposta).
        # Per pulizia, mostriamo solo la parte dopo "Answer:"
        if "Answer:" in full_response:
            clean_answer = full_response.split("Answer:")[-1].strip()
        else:
            # Caso raro in cui il modello non segue il formato, mostriamo tutto ma rimuoviamo la domanda
            clean_answer = full_response.replace(formatted_prompt, "").strip()
            
        print(f"Answer: {clean_answer}")

    print("END")