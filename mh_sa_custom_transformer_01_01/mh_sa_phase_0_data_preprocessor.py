"""
This script is a Data Pre-processor. Its specific job is to convert your massive collection of text files
(Wikipedia + Books) into a single, highly optimized Binary File (.bin).

It acts as the bridge between your raw data (text) and the high-speed training script (numbers).

Here is a breakdown of exactly what it does and why it is superior to the previous methods:

    Input: Thousands of text files (Slow to open, messy encodings).

    Output: One single file of pure numbers (Instant to load, crash-proof).

    You must run this script once before you can run the "Fast Memory Map" training script.

Ignore The Warning
    The tokenizer class you load from this checkpoint is 'GPT2Tokenizer'. The class this function is called from is 'PreTrainedTokenizerFast'.
    Because You trained a custom BPE tokenizer and saved it. When you load it back using the generic PreTrainedTokenizerFast, Hugging Face just notes that the config file inside mentions "GPT2".
"""


import os
import glob
import numpy as np
from tqdm import tqdm  # pip install tqdm
from mh_sa_algorithms_for_custom_transformer_model import load_liver_tokenizer
from datetime import datetime

# --- CONFIGURATION ---
OUTPUT_FILE = "bin_dataset/wiki_books_dataset.bin"
DATA_FOLDERS = [
    "../../../Datasets/Txt_Books/",
    "../../../Datasets/Au_Books/",
    "../../../Datasets/WikipediaDump/Final_Training_Data"
]


def preprocess_data():
    print("\npreprocess_data() - BEGIN")

    tokenizer, vocab_size = load_liver_tokenizer()
    # --- SAFETY CHECK FOR UINT16 ---
    # uint16 holds max 65,535. If your vocab is larger (e.g. 100k), this corrupts data.
    if vocab_size > 65535:
        print(f"❌ Error: Vocabulary size ({vocab_size}) is too large for uint16.")
        print("Please change dtype to np.uint32 in the script.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 1. Gather all files
    files = []
    for folder in DATA_FOLDERS:
        # Recursive search for .txt
        found = sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
        files.extend(found)

    print(f"Found {len(files)} text files. Starting tokenization...")

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
    print("preprocess_data() - END\n")



if __name__ == "__main__":
    print("\n__main__() - BEGIN")
    begin_time = datetime.now()
    preprocess_data()
    print(f"Elapsed: {datetime.now() - begin_time}")
    print("__main__() - END\n")
