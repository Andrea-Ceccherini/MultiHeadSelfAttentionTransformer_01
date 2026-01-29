import os


def folder_analysis(folder_paths):
    """
    Analizza una lista di percorsi cartella.
    Ritorna un dizionario con conteggio file e dimensione totale in byte.
    """
    analysis_results = {}

    for path in folder_paths:
        # Verifichiamo se il percorso esiste ed è effettivamente una cartella
        if not os.path.exists(path):
            analysis_results[path] = "Errore: Percorso non trovato"
            continue
        if not os.path.isdir(path):
            analysis_results[path] = "Errore: Il percorso non è una cartella"
            continue

        file_count = 0
        total_bytes = 0

        # os.walk attraversa la cartella e tutte le sue sottocartelle
        for root, dirs, files in os.walk(path):
            for name in files:
                file_path = os.path.join(root, name)

                # Gestiamo eventuali errori di permessi per singoli file
                try:
                    # skip_symlinks: opzionale, qui contiamo il file reale
                    if not os.path.islink(file_path):
                        total_bytes += os.path.getsize(file_path)
                        file_count += 1
                except (OSError, PermissionError):
                    # Salta i file che non possono essere letti
                    continue

        analysis_results[path] = {
            "file_count": file_count,
            "total_bytes": total_bytes
        }

    return analysis_results

# Esempio di utilizzo:
cartelle_da_testare = ["/home/andrea/Datasets/LiverDataset", "/home/andrea/Datasets/Txt_WikipediaData", "/home/andrea/Datasets/Txt_Au_Books", "/home/andrea/Datasets/Txt_Books/", "/home/andrea/Datasets/Txt_WikipediaData/", "/home/andrea/Datasets/WikipediaDump/Txt_Final_Training_Data/"]
report = folder_analysis(cartelle_da_testare)
print(report)

# Assuming 'report' is the dictionary returned by folder_analysis

print("--- FOLDER ANALYSIS REPORT ---")
for folder, data in report.items():
    if isinstance(data, str):
        # This handles the error messages we built into the function
        print(f"Directory: {folder} | Status: {data}")
    else:
        print(f"Directory: {folder}")
        print(f"  - Files found: {data['file_count']}")
        print(f"  - Total Size:  {data['total_bytes']} bytes")
    print("-" * 30)