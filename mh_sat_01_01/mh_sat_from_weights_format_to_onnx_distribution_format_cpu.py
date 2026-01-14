import torch
import os
from safetensors.torch import load_file

# --- CPU MODE (Safe) ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mh_sat_algorithms_for_custom_transformer_model import (
    CustomTransformer, load_gpt2_tokenizer,
    TOKENIZATION_MAX_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, DROPOUT
)

# --- CONFIGURATION ---
INPUT_WEIGHTS = "supervised_model_weights/fine_tuned_best.safetensors"
OUTPUT_ONNX = "../onnx_model_distribution_format/my_liver_model.onnx"  #
DIST_FOLDER = "../onnx_model_distribution_format/"

def export_to_onnx():
    print(f"--- Exporting to ONNX: {OUTPUT_ONNX} ---")

    if not os.path.exists(INPUT_WEIGHTS):
        print(f"❌ Error: {INPUT_WEIGHTS} not found.")
        return

    # 1. Load Model
    print("Loading Model...")
    _, vocab_size = load_gpt2_tokenizer()
    model = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    model.load_state_dict(load_file(INPUT_WEIGHTS))
    model.eval()

    # 2. Create Dummy Input
    # ONNX needs to "trace" the model execution once to understand the graph
    print("Tracing model graph...")
    dummy_src = torch.randint(0, vocab_size, (1, TOKENIZATION_MAX_LENGTH))  # Batch 1
    dummy_tgt = torch.randint(0, vocab_size, (1, TOKENIZATION_MAX_LENGTH))

    # Ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_ONNX), exist_ok=True)

    # 3. Export
    torch.onnx.export(
        model,
        (dummy_src, dummy_tgt),  # The inputs your forward() function expects
        OUTPUT_ONNX,
        export_params=True,
        opset_version=18,  # Match PyTorch Nightly features
        do_constant_folding=True,
        input_names=['encoder_input', 'decoder_input'],
        output_names=['logits'],
        dynamic_axes={
            'encoder_input': {0: 'batch_size', 1: 'seq_len'},
            'decoder_input': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'}
        }
    )

    print(f"✅ Success! ONNX model saved to: {OUTPUT_ONNX}")
    print(f"   Size: {os.path.getsize(OUTPUT_ONNX) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    tokenizer, _ = load_gpt2_tokenizer()
    tokenizer._tokenizer.save(os.path.join(DIST_FOLDER, "tokenizer.json"))
    print("✅ Tokenizer JSON saved.")
    export_to_onnx()