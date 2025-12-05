import torch
import torch.nn.functional as F
import os
import sys

# --- AMD/ROCm SETUP ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)

from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    CustomTransformer, load_gpt2_tokenizer,
    TOKENIZATION_MAX_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, DROPOUT
)
from safetensors.torch import load_file

# --- CONFIGURATION ---
# MODEL_PATH = "supervised_qa_model_files/fine_tuned_best.safetensors"
MODEL_PATH = "unsupervised_model_weights/latest_checkpoint.safetensors"  # for test purpose: Check the Foundation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Loading Tokenizer...")
    tokenizer, vocab_size = load_gpt2_tokenizer()

    print(f"Loading Model Architecture...")
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

    print(f"Loading Weights from {MODEL_PATH}...")
    state_dict = load_file(MODEL_PATH)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

    print(f"\n🔍 Generating from: '{prompt}'")

    for i in range(max_new_tokens):
        cond_ids = input_ids[:, -TOKENIZATION_MAX_LENGTH:]

        with torch.no_grad():
            outputs = model(cond_ids, cond_ids)
            next_token_logits = outputs[:, -1, :]

            # --- FORCE TALK FIX ---
            # Manually set the probability of EOS (50256) to negative infinity
            # This makes it impossible for the model to stop.
            if tokenizer.eos_token_id is not None:
                next_token_logits[:, tokenizer.eos_token_id] = -float('inf')

            # Temperature
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # Debug: Print tokens as they appear
            word = tokenizer.decode(next_token_id[0])
            print(f"   Step {i + 1}: {word} (ID: {next_token_id.item()})")

    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return full_text


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("✅ Model Loaded. Ready to chat.\n")

    # Test Prompts
    test_questions = [
        "What is the function of the liver?",
        "What are the symptoms of hepatitis?",
        "The capital of France is",  # General knowledge test
        "Explain bile production."
    ]

    print("--- AUTOMATIC TESTS ---")
    for q in test_questions:
        print(f"\n❓ Input: {q}")
        # We assume the dataset format was "Question: ... Answer: ..." or similar.
        # Let's prompt it slightly to encourage an answer.
        # If your CSV just had raw text, standard prompting works.
        response = generate_text(model, tokenizer, q, max_new_tokens=60)
        print(f"🤖 Output: {response}")
        print("-" * 30)

    print("\n--- INTERACTIVE MODE ---")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = generate_text(model, tokenizer, user_input, max_new_tokens=100)
        print(f"Bot: {response}")