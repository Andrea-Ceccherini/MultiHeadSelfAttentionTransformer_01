import os
import glob
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast

# --- CONFIGURATION ---
OUTPUT_DIR = "custom_tokenizer_files"
VOCAB_SIZE = 50258
MIN_FREQUENCY = 2

DATA_FOLDERS = [
    "../../../Datasets/Txt_Books/",
    "../../../Datasets/Au_Books/",
    "../../../Datasets/WikipediaDump/Final_Training_Data"
]


def get_file_list():
    files = []
    for folder in DATA_FOLDERS:
        found = sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
        files.extend(found)
    return files


# --- RAM-SAFE GENERATOR ---
def corpus_iterator(files):
    total_files = len(files)
    for i, file_path in enumerate(files):
        # Print progress slightly more often to catch where it crashes
        if (i + 1) % 50 == 0:
            print(f"   Processing file {i + 1}/{total_files}...")

        try:
            # 'errors=replace' fixes encoding crashes
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                # OPTIMIZATION: Read line by line instead of f.read()
                # This prevents loading a 1GB Wikipedia chunk into RAM all at once.
                for line in f:
                    text = line.strip()
                    if text:
                        yield text

        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            continue


def train_tokenizer():
    files = get_file_list()
    print(f"Found {len(files)} files. Starting RAM-safe training...")

    tokenizer = ByteLevelBPETokenizer()

    # Train
    tokenizer.train_from_iterator(
        corpus_iterator(files),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        show_progress=True,
        special_tokens=["<|endoftext|>", "<|padding|>"]
    )

    # Save Raw Files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer.save_model(OUTPUT_DIR)
    print("✅ Raw tokenizer saved.")

    # Wrap in Hugging Face Format
    hf_tokenizer = GPT2TokenizerFast(
        vocab_file=os.path.join(OUTPUT_DIR, "vocab.json"),
        merges_file=os.path.join(OUTPUT_DIR, "merges.txt"),
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>"
    )

    hf_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"🎉 Custom Tokenizer ready in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    train_tokenizer()