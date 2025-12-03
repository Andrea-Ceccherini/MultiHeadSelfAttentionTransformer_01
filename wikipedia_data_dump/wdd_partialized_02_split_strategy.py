import subprocess
import os
import sys
import json
import bz2
from datetime import datetime
import glob

"""
WIKIEXTRACTOR SPLIT & RESUME SCRIPT (FIXED FOR VENV)

Strategy (Option A):
1. PRE-PROCESSING: Splits the dump if not already split.
2. EXECUTION: Processes split files using 'python -m wikiextractor.WikiExtractor'.
3. RESUME: Tracks progress in JSON.
"""

# --- Global Configuration ---

INPUT_FILE = r"../../../Datasets/WikipediaDump/enwiki-latest-pages-articles.xml.bz2"
SPLIT_DIR = r"../../../Datasets/WikipediaDump/Split_Chunks"
OUTPUT_DIR = r"../../../Datasets/WikipediaDump/Extracted_Text_Corpus"
STATE_FILE = os.path.join(os.path.dirname(INPUT_FILE), "wikiextractor_progress_split.json")
CHUNK_TARGET_SIZE = 500 * 1024 * 1024


# --- Phase 1: Splitter Logic ---

def get_xml_header(file_path):
    header = []
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            header.append(line)
            if '<page>' in line:
                header.pop()
                break
    return "".join(header)


def split_dump_if_needed():
    # Check if we already have chunks
    if os.path.exists(SPLIT_DIR) and len(glob.glob(os.path.join(SPLIT_DIR, "chunk_*.xml"))) > 0:
        print(f"✅ Split files found in {SPLIT_DIR}. Skipping split process.")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    print(f"⚡ Starting Split Process (This happens only once)...")
    os.makedirs(SPLIT_DIR, exist_ok=True)

    header_content = get_xml_header(INPUT_FILE)
    footer_content = "</mediawiki>"

    chunk_counter = 1
    current_size = 0
    current_file_path = os.path.join(SPLIT_DIR, f"chunk_{chunk_counter:03d}.xml")

    out_file = open(current_file_path, 'w', encoding='utf-8')
    out_file.write(header_content)

    print(f"   --> Writing {os.path.basename(current_file_path)}...")

    with bz2.open(INPUT_FILE, 'rt', encoding='utf-8') as infile:
        inside_page = False
        for line in infile:
            if '<page>' in line: inside_page = True
            out_file.write(line)
            current_size += len(line.encode('utf-8'))

            if '</page>' in line:
                inside_page = False
                if current_size >= CHUNK_TARGET_SIZE:
                    out_file.write("\n" + footer_content)
                    out_file.close()
                    chunk_counter += 1
                    current_size = 0
                    current_file_path = os.path.join(SPLIT_DIR, f"chunk_{chunk_counter:03d}.xml")
                    print(f"   --> Writing {os.path.basename(current_file_path)}...")
                    out_file = open(current_file_path, 'w', encoding='utf-8')
                    out_file.write(header_content)

    if not out_file.closed:
        out_file.write("\n" + footer_content)
        out_file.close()
    print("✅ Splitting Complete.")


# --- Phase 2: State Management ---

def load_progress():
    split_files = sorted(glob.glob(os.path.join(SPLIT_DIR, "chunk_*.xml")))
    if not split_files:
        print("❌ Error: No split files found.")
        sys.exit(1)

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            saved_state = json.load(f)
        if len(saved_state["chunks"]) == len(split_files):
            print(f"Loading progress from: {STATE_FILE}")
            return saved_state
        else:
            print("⚠️ Warning: File list changed. Resetting state.")

    print("Creating initial state map based on split files.")
    chunks = []
    for fpath in split_files:
        chunks.append({
            "file_path": fpath,
            "filename": os.path.basename(fpath),
            "completed": False
        })
    return {"chunks": chunks}


def save_progress(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)
    print(f"   (State saved)")


# --- Phase 3: Execution ---

def run_wikiextractor_managed():
    split_dump_if_needed()
    state = load_progress()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunks = state["chunks"]
    total_chunks = len(chunks)

    for index, chunk in enumerate(chunks):
        if chunk["completed"]:
            continue

        fpath = chunk["file_path"]
        fname = chunk["filename"]

        print(f"\n--- Processing Block {index + 1}/{total_chunks}: {fname} ---")

        chunk_subfolder_name = os.path.splitext(fname)[0]
        current_output_dir = os.path.join(OUTPUT_DIR, chunk_subfolder_name)
        os.makedirs(current_output_dir, exist_ok=True)

        start_time = datetime.now()

        # --- THE FIX IS HERE ---
        # We use sys.executable (the current python) to call the module
        command = [
            sys.executable,  # Points to your .venv/bin/python
            "-m", "wikiextractor.WikiExtractor",  # Invoke as a module
            fpath,
            "--output", current_output_dir,
            "--processes", str(os.cpu_count() or 1),
            "--bytes", "10M",
            "--quiet"
        ]
        # -----------------------

        try:
            print(f"   Executing WikiExtractor on {fname}...")
            subprocess.run(command, check=True)

            chunk["completed"] = True
            save_progress(state)

            elapsed = datetime.now() - start_time
            print(f"✅ Completed {fname} in {elapsed}")

        except subprocess.CalledProcessError:
            print(f"\n❌ ERROR: WikiExtractor failed on {fname}.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user.")
            sys.exit(0)

    print("\n🎉 ALL BLOCKS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_wikiextractor_managed()