"""
This script is a Data Pre-processor. Its specific job is to convert your massive collection of text files
(Wikipedia + Books) into a single, highly optimized Binary File (.bin).

It acts as the bridge between your raw data (text) and the high-speed training script (numbers).

Here is a breakdown of exactly what it does and why it is superior to the previous methods:

    Input: Thousands of text files (Slow to open, messy encodings).

    Output: One single file of pure numbers (Instant to load, crash-proof).

    You must run this script once before you can run the "Fast Memmap" training script.
"""


import os
import glob
import numpy as np
from tqdm import tqdm  # pip install tqdm
from mh_sat_algorithms_for_custom_transformer_model import load_gpt2_tokenizer

# --- CONFIGURATION ---
OUTPUT_FILE = "../wiki_books_dataset/wiki_books_dataset.bin"
DATA_FOLDERS = [
    "../../../Datasets/Txt_Books/",
    "../../../Datasets/Au_Books/",
    "../../../Datasets/WikipediaDump/Final_Training_Data"
]


def preprocess_data():
    tokenizer, vocab_size = load_gpt2_tokenizer()

    # 1. Gather all files
    files = []
    for folder in DATA_FOLDERS:
        # Recursive search for .txt
        found = sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
        files.extend(found)

    print(f"Found {len(files)} text files. Starting tokenization...")

    # 2. We will write to a binary file incrementally
    # GPT2 vocab is ~50257, so it fits in uint16 (0-65535). This saves 50% RAM/Disk vs int32.
    token_count = 0

    # Open file in binary write mode
    with open(OUTPUT_FILE, "wb") as f:
        for file_path in tqdm(files, desc="Processing Files"):
            try:
                # errors='replace' fixes the book crashes
                with open(file_path, "r", encoding="utf-8", errors="replace") as txt_f:
                    text = txt_f.read()

                if not text: continue

                # Tokenize (fastest way: encode raw string)
                # We add the EOS token at the end of every file so the model knows where one doc ends
                tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]

                # Convert to numpy uint16
                arr = np.array(tokens, dtype=np.uint16)

                # Write raw bytes to disk
                f.write(arr.tobytes())
                token_count += len(tokens)

            except Exception as e:
                print(f"Skipped {file_path}: {e}")

    print(f"\n✅ Done! Saved {token_count:,} tokens to {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / (1024 ** 3):.2f} GB")


if __name__ == "__main__":
    preprocess_data()