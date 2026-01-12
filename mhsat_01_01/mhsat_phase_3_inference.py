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
MODEL_PATH = "../mh_sat_01_01/unsupervised_model_weights/phase1_FINAL_3.77.safetensors"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("MODEL_PATH = ", MODEL_PATH)

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


def generate_text(model, tokenizer, question, max_new_tokens=100, temperature=0.6):
    # 1. Format the input EXACTLY like Phase 2 training
    # The model expects "Question: ... Answer:"
    prompt = f"Question: {question} Answer:"

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

    print(f"\n🔍 Input Prompt: '{prompt}'")

    # Generate loop
    for i in range(max_new_tokens):
        cond_ids = input_ids[:, -TOKENIZATION_MAX_LENGTH:]

        with torch.no_grad():
            outputs = model(cond_ids, cond_ids)
            next_token_logits = outputs[:, -1, :]

            # --- CRITICAL CHANGE: ALLOW EOS ---
            # We removed the block that set EOS to -inf.
            # Now the model is allowed to stop.

            # Temperature
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample
            next_token_id = torch.multinomial(probs, num_samples=1)

            # --- STOPPING CONDITIONS ---
            # 1. If it predicts EOS, stop.
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # 2. Check if it generated a new "Question:" tag (Hallucination loop)
            # This is a simple heuristic to stop it from rambling
            current_text = tokenizer.decode(input_ids[0])
            if "Question:" in current_text[len(prompt):]:
                # If "Question:" appears AFTER our prompt, stop.
                break

    # Decode and clean up
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Extract just the answer part
    if "Answer:" in full_text:
        answer = full_text.split("Answer:")[1].strip()
        # Clean up any trailing "Question:" if the loop didn't catch it
        answer = answer.split("Question:")[0].strip()
        return answer
    else:
        return full_text


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("✅ Model Loaded. Ready to chat.\n")

    test_questions = [
        "What is the function of the liver?",
        "What are the symptoms of hepatitis?",
        "The capital of France is",
    ]

    print("--- AUTOMATIC TESTS ---")
    for q in test_questions:
        response = generate_text(model, tokenizer, q)
        print(f"🤖 Answer: {response}")
        print("-" * 30)

    print("\n--- INTERACTIVE MODE ---")
    while True:
        user_input = input("\nAsk about Liver (or anything): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = generate_text(model, tokenizer, user_input)
        print(f"Bot: {response}")