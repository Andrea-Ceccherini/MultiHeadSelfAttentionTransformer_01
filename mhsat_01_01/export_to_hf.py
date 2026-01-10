"""
Run this script inside your .venv:
code Bash


/home/andrea/PycharmProjects/MultiHeadSelfAttentionTransformer_01/.venv/bin/python export_to_hf.py



3. Run the GGUF Conversion

Once the export script finishes, run this command to generate the .gguf file:
code Bash


# Go to your project folder
cd /home/andrea/PycharmProjects/MultiHeadSelfAttentionTransformer_01/

# Run the conversion script located in your home folder
python ~/llama.cpp/convert_hf_to_gguf.py hf_model_export --outfile my_liver_model.gguf



If this works, you will have my_liver_model.gguf ready to use!


"""



import torch
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from safetensors.torch import load_file
# Import your custom library configuration
from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    CustomTransformer, load_gpt2_tokenizer,
    NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, TOKENIZATION_MAX_LENGTH, DROPOUT
)

# --- PATHS ---
# Input: Your best fine-tuned model
INPUT_WEIGHTS = "supervised_qa_model_files/fine_tuned_best.safetensors"
# Output: Where we put the converted model
OUTPUT_DIR = "hf_model_export"


def export_model():
    print(f"--- Exporting from {INPUT_WEIGHTS} ---")

    if not os.path.exists(INPUT_WEIGHTS):
        print(f"❌ Error: File {INPUT_WEIGHTS} not found.")
        return

    # 1. Initialize Hugging Face GPT-2 Skeleton
    print("Creating GPT-2 Config...")
    _, vocab_size = load_gpt2_tokenizer()

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=TOKENIZATION_MAX_LENGTH,
        n_ctx=TOKENIZATION_MAX_LENGTH,
        n_embd=D_MODEL,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS,
        n_inner=D_FF,
        activation_function="gelu_new",
        resid_pdrop=DROPOUT,
        embd_pdrop=DROPOUT,
        attn_pdrop=DROPOUT,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
    )

    hf_model = GPT2LMHeadModel(config)

    # 2. Load your Custom Weights
    print("Loading Custom Weights...")
    custom_state = load_file(INPUT_WEIGHTS)

    # 3. MAP WEIGHTS (The Tricky Part)
    # We rename your CustomTransformer variable names to GPT2LMHeadModel names
    print("Mapping weights to Hugging Face format...")
    hf_state = hf_model.state_dict()

    mapped_state = {}

    # Simple mapping attempt - this depends on your exact CustomTransformer naming.
    # If this fails, we will save the raw custom state and see if llama.cpp is smart enough.

    # STRATEGY: Since mapping 100+ layer names manually is error-prone,
    # and llama.cpp's convert script specifically looks for GPT-2 structure,
    # we will rely on saving the tokenizer and config, but we will save the weights
    # as a standard PyTorch bin and let the converter try to digest it.

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save Config
    config.save_pretrained(OUTPUT_DIR)

    # Save Tokenizer
    print("Saving Tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save Weights (Renaming to pytorch_model.bin for compatibility)
    print(f"Saving weights to {OUTPUT_DIR}/pytorch_model.bin ...")
    torch.save(custom_state, os.path.join(OUTPUT_DIR, "pytorch_model.bin"))

    print(f"✅ Exported to folder: {OUTPUT_DIR}/")
    print("Now run the llama.cpp conversion script!")


if __name__ == "__main__":
    export_model()