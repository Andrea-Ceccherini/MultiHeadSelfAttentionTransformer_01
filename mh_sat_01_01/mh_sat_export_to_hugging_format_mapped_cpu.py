import torch
import os
import json
from transformers import GPT2Config, GPT2TokenizerFast
from safetensors.torch import load_file

# --- CONFIGURATION ---
INPUT_WEIGHTS = "supervised_model_weights/fine_tuned_best.safetensors"
OUTPUT_DIR = "hugging_face_model_distribution_format"

# Use CPU for safety
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mh_sat_algorithms_for_custom_transformer_model import (
    load_gpt2_tokenizer,
    NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, TOKENIZATION_MAX_LENGTH, DROPOUT
)


def rename_key(key):
    """
    Translates CustomTransformer variable names to Hugging Face GPT-2 names.
    This mapping depends on your specific CustomTransformer implementation.
    """
    # 1. Embeddings
    if "encoder_embedding" in key:
        return key.replace("encoder_embedding", "transformer.wte")
    if "pos_embedding" in key or "position_embedding" in key:
        return "transformer.wpe.weight"

    # 2. Layers / Blocks
    # Custom usually uses "blocks.0", HF uses "transformer.h.0"
    if "blocks" in key or "layers" in key:
        new_key = key.replace("blocks", "transformer.h").replace("layers", "transformer.h")

        # Attention parts
        # HF GPT-2 uses Conv1D, so weights are combined.
        # If your model separates Q, K, V, this gets tricky.
        # Assuming your model uses a standard Linear c_attn:
        new_key = new_key.replace("self_attention.c_attn", "attn.c_attn")
        new_key = new_key.replace("self_attention.c_proj", "attn.c_proj")

        # Feed Forward
        new_key = new_key.replace("feed_forward.c_fc", "mlp.c_fc")
        new_key = new_key.replace("feed_forward.c_proj", "mlp.c_proj")

        # Layer Norms
        # Custom often has norm1/norm2. HF uses ln_1/ln_2
        new_key = new_key.replace("norm1", "ln_1")
        new_key = new_key.replace("norm2", "ln_2")

        return new_key

    # 3. Final Layer Norm
    if "layer_norm" in key:
        return key.replace("layer_norm", "transformer.ln_f")

    # 4. Output Head (Language Model Head)
    # Often shares weights with encoder_embedding in GPT-2, but if separate:
    if "lm_head" in key:
        return key  # HF expects lm_head.weight usually

    return key  # Return original if no match found


def export_model():
    print(f"--- Processing {INPUT_WEIGHTS} ---")

    if not os.path.exists(INPUT_WEIGHTS):
        print(f"❌ Error: {INPUT_WEIGHTS} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. CREATE CONFIG (With Explicit Architecture Tags)
    print("Generating Config...")
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
        # THESE ARE THE MISSING KEYS causing your error:
        model_type="gpt2",
        architectures=["GPT2LMHeadModel"]
    )
    config.save_pretrained(OUTPUT_DIR)

    # 2. SAVE TOKENIZER
    print("Saving Tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 3. LOAD AND RENAME WEIGHTS
    print("Loading Custom Weights...")
    custom_state = load_file(INPUT_WEIGHTS)

    print("Remapping keys to Hugging Face GPT-2 format...")
    hf_state = {}

    for old_key, tensor in custom_state.items():
        new_key = rename_key(old_key)

        # Handle Transpose for Conv1D layers if necessary
        # HF GPT-2 uses Conv1D for Linear layers, which stores weights as (Input, Output)
        # PyTorch Linear stores as (Output, Input).
        # Depending on your CustomTransformer implementation, we might need `.t()`
        # Usually, if you used nn.Linear, and HF expects Conv1D, we transpose.
        if "c_attn.weight" in new_key or "c_proj.weight" in new_key or "c_fc.weight" in new_key:
            # Check shapes. If Custom is (Out, In) and HF wants (In, Out)
            # For now, we pass it as is. If llama.cpp complains about shape, we uncomment .t()
            # tensor = tensor.t()
            pass

        hf_state[new_key] = tensor
        if old_key != new_key:
            print(f"  Mapped: {old_key} -> {new_key}")

    # 4. SAVE BINARY
    output_bin = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
    print(f"Saving re-mapped weights to {output_bin}...")
    torch.save(hf_state, output_bin)

    print(f"✅ Ready for conversion in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    export_model()