import os
import json
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

# --- RDNA 4 OPTIMIZATION: Clean Environment ---
# Ensure no debug flags slow us down or force serial execution
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

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

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_dir = "supervised_qa_model_files"
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Did you run Phase 2?")
        exit()

    # 1. Load Config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    # 2. Load Tokenizer
    tokenizer = create_tokenizer_safe(model_dir)

    # 3. Instantiate Model Structure
    print("Initializing model architecture...")
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
    # We look for model.safetensors (saved at end of Phase 2)
    weights_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "model_best.safetensors")

    print(f"Loading weights from {weights_path}")
    state_dict = load_file(weights_path)

    # --- FIX: STRIP '_orig_mod.' PREFIX ---
    # Because Phase 2 used torch.compile, keys have "_orig_mod." added to them.
    # We must remove this prefix to load them into a standard model structure.
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
        else:
            new_key = key
        new_state_dict[new_key] = value

    try:
        model.load_state_dict(new_state_dict)
        print("Weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Tip: Check if your library file has the updated Type-Safe mask functions.")
        exit()

    # 5. RDNA 4 OPTIMIZATION: Convert to FP16
    # This makes inference lightning fast on RX 9070 XT
    model.to(device)
    print("Converting model to Half Precision (FP16)...")
    model.half()

    print("\n✅ Model Ready! The bot is listening.")
    print("------------------------------------------------")
    print("Note: The model expects questions about the Liver or definitions.")
    print("------------------------------------------------")

    # 6. Inference Loop
    while True:
        try:
            user_input = input("\nQuestion (type 'exit' to quit): ")
            if user_input.lower() in ['exit', 'quit']:
                break

            if len(user_input.strip()) < 2:
                continue

            # --- PROMPT FORMATTING ---
            # We must match the format used in training: "Question: ... \nAnswer:"
            formatted_prompt = f"Question: {user_input}\nAnswer:"

            # --- GENERATION ---
            full_response = generate_text_with_beam(
                model, tokenizer, formatted_prompt,
                max_output_length=60,  # Keep it short to avoid hallucinations
                beam_width=5,  # Higher = Better quality
                temperature=0.5  # 0.5 is safer/less random than 0.7
            )

            # --- CLEANUP ---
            if "Answer:" in full_response:
                clean_answer = full_response.split("Answer:")[-1].strip()
            else:
                clean_answer = full_response.replace(formatted_prompt, "").strip()

            print(f"Answer: {clean_answer}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during generation: {e}")

    print("END")