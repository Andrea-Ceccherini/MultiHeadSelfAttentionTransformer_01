import subprocess
import os
import sys
import json
from datetime import datetime

"""
Questo script definisce $10$ blocchi di ID, esegue il primo blocco non completato e aggiorna il file di stato. 
Se il lancio fallisce, riparte dal blocco incompleto.
Non potendo conoscere gli ID esatti dei $18.7$ milioni di titoli senza analizzare il dump, si definisce un intervallo 
di ID ragionevolmente grande per la suddivisione. Usiamo un sistema di suddivisione in blocchi numerici (chunks) e 
salvataggio dello stato in un file JSON.

Istruzioni per l'Uso
Lancio Iniziale: Esegui lo script: python il_tuo_script.py.

Lo script creerà il file wikiextractor_progress.json e lancerà il Blocco 1 (ID 0 a circa 9 milioni).

Lancio Successivo (Continua): Esegui lo script una seconda volta.

Lo script leggerà il file JSON, vedrà che il Blocco 1 è completed: true, e lancerà automaticamente il Blocco 2.

Lancio Fallito (Ripristino): Se il Blocco 3 fallisce, l'eccezione viene catturata, ma il file JSON non viene aggiornato.

Al lancio successivo, lo script vedrà che il Blocco 3 è ancora completed: false e riproverà a eseguire il Blocco 3 
dall'inizio.

Questo metodo garantisce che, anche in caso di interruzione, il tuo costoso processo di parsing riprenda esattamente 
dal punto in cui si è verificato il problema.
"""



# --- Configurazione Globale ---
INPUT_FILE = r"C:\Users\andre\Datasets\WikipediaDump\enwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR = r"C:\Users\andre\Datasets\WikipediaDump\Extracted_Text_Corpus"
STATE_FILE = os.path.join(os.path.dirname(INPUT_FILE), "wikiextractor_progress.json")
NUM_CHUNKS = 10  # Il numero di parti in cui suddividere il lavoro

# Stima gli ID massimi della Wikipedia in inglese (sono nell'ordine delle decine di milioni)
# Stima approssimativa dell'ID massimo della Wikipedia in inglese (circa 90 milioni per le versioni recenti)
MAX_WIKI_ID = 90000000


# --- Funzioni di Gestione dello Stato (Checkpoint) ---

def load_progress():
    """Carica lo stato di avanzamento da un file JSON o crea uno stato iniziale."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            print(f"Caricamento stato da: {STATE_FILE}")
            return json.load(f)

    print("Creazione stato iniziale.")
    # Crea i blocchi di lavoro (Chunks) in base all'ID.
    chunk_size = MAX_WIKI_ID // NUM_CHUNKS
    chunks = []

    for i in range(NUM_CHUNKS):
        min_id = i * chunk_size
        max_id = (i + 1) * chunk_size - 1
        if i == NUM_CHUNKS - 1:
            max_id = MAX_WIKI_ID  # Assicura che l'ultimo blocco copra l'ID massimo

        chunks.append({
            "id": i + 1,
            "min_id": min_id,
            "max_id": max_id,
            "completed": False
        })

    return {"chunks": chunks}


def save_progress(state):
    """Salva lo stato di avanzamento nel file JSON."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)
    print(f"Stato salvato: {STATE_FILE}")


# --- Esecuzione di WikiExtractor ---

def run_wikiextractor_chunked():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERRORE: File di input non trovato: {INPUT_FILE}")
        sys.exit(1)

    state = load_progress()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Trova il prossimo blocco da eseguire
    chunk_to_run = next((chunk for chunk in state["chunks"] if not chunk["completed"]), None)

    if not chunk_to_run:
        print("✅ Tutti i blocchi sono già stati completati. Terminato.")
        return

    chunk_id = chunk_to_run["id"]
    min_id = chunk_to_run["min_id"]
    max_id = chunk_to_run["max_id"]

    print(f"\n--- AVVIO Blocco {chunk_id}/{NUM_CHUNKS} ---")
    print(f"ID Range: da {min_id} a {max_id}")

    # 2. Imposta l'output specifico per questo blocco
    # Creiamo una sottocartella per ogni blocco per evitare confusione
    chunk_output_dir = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id}")
    os.makedirs(chunk_output_dir, exist_ok=True)

    start_time = datetime.now()

    # 3. Costruzione del Comando con i parametri di interruzione
    command = [
        "wikiextractor",
        INPUT_FILE,
        "--output", chunk_output_dir,
        "--min-id", str(min_id),  # Inizia da questo ID
        "--max-id", str(max_id),  # Termina a questo ID
        "--processes", str(os.cpu_count() or 1),
        "--bytes", "10M",
        "--quiet"
    ]

    try:
        # Esecuzione
        subprocess.run(command, check=True, capture_output=False)

        # 4. Successo: Aggiorna lo stato come completato
        chunk_to_run["completed"] = True
        save_progress(state)

        end_time = datetime.now()
        elapsed_time = end_time - start_time

        print(f"\n✅ Blocco {chunk_id} completato con successo in {elapsed_time}.")
        print(f"Il prossimo lancio ripartirà dal blocco {chunk_id + 1}.")

    except subprocess.CalledProcessError:
        print(f"\n❌ ERRORE: Blocco {chunk_id} fallito. Lo stato NON è stato aggiornato.")
        print("Al prossimo lancio, riproverà a eseguire lo stesso blocco.")
    except FileNotFoundError:
        print("\n❌ ERRORE: Assicurati che 'wikiextractor' sia installato.")


if __name__ == "__main__":
    run_wikiextractor_chunked()