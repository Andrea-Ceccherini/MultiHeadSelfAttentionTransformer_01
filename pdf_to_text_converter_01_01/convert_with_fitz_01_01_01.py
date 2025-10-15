import re
import fitz # PyMuPDF


def pulisci_testo_per_pretraining(testo_grezzo):
    """
    Pulisce il testo estratto dal PDF per renderlo adatto al pre-addestramento di un LLM.
    Questa versione è aggressiva e include regex flessibili per blocchi strutturati,
    intestazioni, piè di pagina e sillabazione.
    """
    # 1. Standardizza gli a capo e gli spazi in eccesso
    testo_pulito = testo_grezzo.replace('\r', '\n')
    testo_pulito = re.sub(r'[ \t]+', ' ', testo_pulito)

    # 2. Rimuovi la sillabazione (parole tagliate con trattino a fine riga)
    # Esempio: 'pre-\naddestramento' -> 'preaddestramento'
    # Cattura un carattere alfanumerico seguito da un trattino, un a capo e un altro carattere alfanumerico.
    testo_pulito = re.sub(r'(\w)-\n(\w)', r'\1\2', testo_pulito)
    # Rimuovi i trattini semplici a fine riga che potrebbero essere rumore o artefatti (se non seguiti da testo)
    testo_pulito = re.sub(r'-\n', r'\n', testo_pulito)

    # 3. Pulizia a livello di linea (rimozione di headers/footers/numeri di pagina)
    lines = testo_pulito.split('\n')
    cleaned_lines = []

    # Pattern per identificare linee di "rumore" che non sono testo informativo
    noise_patterns = [
        # Numeri di pagina e riferimenti semplici (Esempio: - 123 -, [1], 46)
        r'^\s*-\s*\d+\s*-\s*$',
        r'^\s*\d+\s*$',
        r'^\s*\[\d+\]\s*$',

        # Didascalie di figure e tabelle (Esempio: Fig. 1.2, Table 3)
        r'^\s*Fig\.?\s*\d+(\.\d+)?.*',
        r'^\s*Tabl?e?\s*\d+(\.\d+)?.*',

        # Acronimi o titoli brevi in sole maiuscole, inclusi quelli con punti e spazi
        # (Esempio: ABC, ACTH, A:CR)
        r'^\s*([A-Z]\s?\.?\s?){2,}$',

        # AGGIUNTA: Linee composte principalmente da numeri, unità di misura e riferimenti di pagina.
        # Questo cattura la maggior parte delle righe con valori di riferimento e p###.
        # Esempio: 130–180g/L p324, 76–96fL p326; p332, <10mg/L p686
        r'^\s*[\d\.\s\-\–><±]+[a-zA-Z\/]{0,5}\s*(p\d+|[\d,\.]{1,2})\s*(\(.*\))?\s*$',

        # AGGIUNTA: Righe che sembrano abbreviazioni seguite da una descrizione,
        # spesso composte da poche parole o solo punteggiatura
        # Esempio: ABG .....arterial blood gas: PaO2, PaCO2, pH, HCO3
        r'^\s*[A-Z]{2,6}\s{2,}\..{1,4}\s+.*(\d{1,3}|\d{1,3}–\d{1,3})$'
    ]

    for line in lines:
        line_stripped = line.strip()
        is_noise = False

        # Le linee molto corte (tra 1 e 50 caratteri) sono i candidati principali al rumore
        if 0 < len(line_stripped) < 50:
            # Controlla se la linea è composta quasi esclusivamente da punteggiatura/numeri/spazi
            if re.sub(r'[\s\d\W]', '', line_stripped) == '':
                is_noise = True

            # Controlla i pattern di rumore specifici
            if not is_noise:
                for pattern in noise_patterns:
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        is_noise = True
                        break

        # Aggiunge solo le linee che non sono rumore e non sono completamente vuote
        if not is_noise and line_stripped:
            cleaned_lines.append(line_stripped)

    testo_pulito = '\n'.join(cleaned_lines)

    # 4. Consolidamento dei paragrafi
    # Sostituisce linee vuote in eccesso con un unico doppio a capo (separatore di paragrafo)
    testo_pulito = re.sub(r'\n\s*\n+', '\n\n', testo_pulito.strip())

    return testo_pulito


def convert_pdf_in_txt(input_pdf_path, output_txt_path):
    """
    Estrae il testo da un file PDF, lo pulisce in modo aggressivo e lo salva in un file .txt.
    """
    print(f"Inizio estrazione da: {input_pdf_path}")
    testo_grezzo = ""
    try:
        # 1. Estrazione del testo grezzo
        doc = fitz.open(input_pdf_path)
        for page in doc:
            # Aggiungi un doppio a capo tra le pagine per facilitare la pulizia della sillabazione
            testo_grezzo += page.get_text() + '\n\n'

        # 2. Pulizia aggressiva
        print("Inizio pulizia aggressiva del testo...")
        testo_pulito = pulisci_testo_per_pretraining(testo_grezzo)

        # 3. Salvataggio del testo pulito
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(testo_pulito)

        print(f"✔ Conversione e pulizia completate. Salvato in: '{output_txt_path}'")

    except FileNotFoundError:
        print(f"❌ Errore: File PDF non trovato al percorso: {input_pdf_path}")
    except Exception as e:
        print(f"❌ Si è verificato un errore durante la conversione/pulizia: {e}")


if __name__ == "__main__":
    print("--- Inizio Processo di Pulizia Testo PDF per LLM ---")
    # I percorsi sono stati mantenuti come nell'esempio, ma dovrebbero essere modificati
    # se eseguiti al di fuori dell'ambiente specificato.
    pdf_document_path = "../../../Datasets/Pdf_Books/Oxford_Handbook_of_Clinical_Medicine_10th_2017_Edition_SamanSarKo.pdf"
    text_document_path = "../../../Datasets/Txt_Books/Oxford_Handbook_of_Clinical_Medicine_10th_2017_Edition_SamanSarKo.txt"

    # Nota sull'ambiente: questo script richiede la libreria 'fitz' (PyMuPDF) per la lettura dei PDF.
    convert_pdf_in_txt(pdf_document_path, text_document_path)

    print("--- Fine Processo ---")
