from pdfminer.high_level import extract_text_to_fp
from io import StringIO
import re



def pulisci_testo_per_pretraining(testo_grezzo):
    """
    Pulisce il testo estratto dal PDF per renderlo adatto al pre-addestramento di un LLM.
    Questa versione è ancora più aggressiva e include regex più flessibili per i blocchi strutturati.
    """
    testo_pulito = testo_grezzo

    # 1. Normalizzazione trattini e apici per semplificare i regex successivi
    testo_pulito = testo_pulito.replace('—', '-')  # Trattino lungo
    testo_pulito = testo_pulito.replace('–', '-')  # Trattino N-dash
    testo_pulito = testo_pulito.replace('‘', "'").replace('’', "'")  # Apici
    testo_pulito = testo_pulito.replace('“', '"').replace('”', '"')  # Virgolette

    # 2. Decodifica ligatures comuni (es. 'ﬂ' -> 'fl', 'ﬁ' -> 'fi')
    testo_pulito = testo_pulito.replace('ﬂ', 'fl').replace('ﬁ', 'fi')

    # 3. Normalizzazione newline: Sostituisce i newline con spazi per ricomporre frasi spezzate.
    testo_pulito = testo_pulito.replace('\n', ' ')

    # 4. Rimozione BLOCCHI DI TESTO STRUTTURATO COMPLETI (utilizzando re.DOTALL)

    # Rimuove la sezione dell'indice "Acute abdomen ... Talleyrand to his coachman." o "Malignant hypertension ..."
    # Si basa sul testo "emergency topics" e la citazione di Talleyrand come punti di ancoraggio.
    testo_pulito = re.sub(
        r'(?:Acute abdomen|Malignant hypertension).*?Index to emergency topics.*?Talleyrand to his coachman\.', ' ',
        testo_pulito, flags=re.DOTALL)
    testo_pulito = re.sub(r'Index to emergency topics.*?Talleyrand to his coachman\.', ' ', testo_pulito,
                          flags=re.DOTALL)

    # Rimuove l'intera sezione "Common haematology values"
    # Aggiusta il pattern finale per catturare 'For all other reference intervals, see –7' o simili
    testo_pulito = re.sub(r'Common haematology values.*?For all other reference intervals, see.*?(\d+\s*-\s*\d+)?', ' ',
                          testo_pulito, flags=re.DOTALL)

    # Rimuove l'intera sezione "Reading tests" (se ricompare)
    testo_pulito = re.sub(r'Reading tests Hold this chart.*?in August\.', ' ', testo_pulito, flags=re.DOTALL)

    # Rimuove la sezione del titolo, autori e copyright esteso
    # Inizia dal pattern del titolo fino a "address above."
    testo_pulito = re.sub(r'OXFORD HANDBOOK OF CLINICALMEDICINETENTH EDITION.*?address above\.', ' ', testo_pulito,
                          flags=re.DOTALL)

    # Rimuove la sezione dei "Translations" (se ricompare)
    testo_pulito = re.sub(
        r'Translations:ChineseFrenchHungarianPolishRussianCzechGermanIndonesianPortugueseSpanishEstonianGreekItalianRomanian',
        ' ', testo_pulito, flags=re.DOTALL)

    # Rimuove la parte di "British Library Cataloguing..." e disclaimer esteso
    testo_pulito = re.sub(
        r'You must not circulate this book in any other binding or cover.*?legal liability for any errors in the text, or for the misuse or misapplication of material in this work\.',
        ' ', testo_pulito, flags=re.DOTALL)
    testo_pulito = re.sub(r'British Library Cataloguing in Publication DataData available.*?ISBN -\d+-\d+-\d+-\d+', ' ',
                          testo_pulito, flags=re.DOTALL)

    # Rimuove l'intera sezione "Contents" (nuovo pattern, più robusto con numeri finali)
    testo_pulito = re.sub(r'ContentsEach chapter’s contents are detailed on its first page.*?Cardiac arrest \d+', ' ',
                          testo_pulito, flags=re.DOTALL)
    testo_pulito = re.sub(
        r'Thinking about medicine.*?ContentsThe Hippocratic oath \d+Medical care \d+Compassion \d+.*?Medicalization Asclepius, the god of healing and his three daughters, Meditrina medicine , Hygieia hygiene , and Panacea healing \.',
        ' ', testo_pulito, flags=re.DOTALL)

    # Rimuove l'intera sezione "Symbols and abbreviations" (ancora più robusto, cercando un pattern finale più generico)
    testo_pulito = re.sub(r'Symbols and abbreviations.*?ZN -Neelsen stain, eg for mycobacteria', ' ', testo_pulito,
                          flags=re.DOTALL)

    # Rimuove la riga "He who studies medicine without books sails an unchartered sea..." e il blocco del paziente.
    testo_pulito = re.sub(
        r'He who studies medicine without books sails an unchartered sea.*?With them, you are a doctor\.', ' ',
        testo_pulito, flags=re.DOTALL)

    # Rimuove la sezione "QALYS and resource rationing" e la tabella/elenco di vantaggi/svantaggi
    testo_pulito = re.sub(
        r'QALYS and resource rationing.*?Distributive justice is the distribution of goods so that those who are worst off become better off \.',
        ' ', testo_pulito, flags=re.DOTALL)

    # Rimuove il blocco "Compassion" e i riferimenti ai libri (Sebastian Faulkes, Milan Kundera, Sophocles)
    testo_pulito = re.sub(
        r'Compassion\d+\s*Sebastian Faulkes, Human Traces, \d+\.\s*Milan Kundera, The Unbearable Lightness of Being, \d+\.\s*Philoctetes by Sophocles BC translation Phillips and Clay, \d+\.',
        ' ', testo_pulito, flags=re.DOTALL)

    # Rimuove la sezione "Preface to the tenth edition" fino a "Preface to the first edition"
    testo_pulito = re.sub(r'Preface to the tenth edition.*?Preface to the first edition', ' ', testo_pulito,
                          flags=re.DOTALL)

    # Rimuove la sezione "Acknowledgements" fino a "3rd-party content."
    testo_pulito = re.sub(
        r'AcknowledgementsHeart-felt thanks to our advisers on specific sections.*?3rd-party content\.', ' ',
        testo_pulito, flags=re.DOTALL)

    # 5. Pulizia di metadati PDF residui ultra-generica
    # Cattura '_OHCM_10e.indb' seguito da qualsiasi cosa non sia una lettera maiuscola per un po' di caratteri
    testo_pulito = re.sub(r'_OHCM_\d+e\.indb [a-z\d\s\:\/\-\–\—\.]*', ' ', testo_pulito)

    # 6. Rimuove URL e stringhe di riferimento residue (es. http, oup.com, 3rd-party)
    testo_pulito = re.sub(
        r'http[s]?://\S+|www\.\S+|(\d+rd-party web addresses|For updates/corrections, see|See for a full list|\.com)',
        ' ', testo_pulito)

    # 7. Rimuove "N." isolati o con numeri e riferimenti a figure residue
    testo_pulito = re.sub(r'\bN\.\s*\d*\b', ' ', testo_pulito)
    testo_pulito = re.sub(r'Fig \.?\s*[A-Z]?\d+\.?\d?', ' ', testo_pulito)
    testo_pulito = re.sub(r'\(fi g \. \d+\s*\)', ' ', testo_pulito)  # Rimuove (fig . )

    # 8. Rimozione aggressiva dei numeri di pagina e riferimenti numerici
    #   - Numeri isolati o con trattini (es. '606', '298', '842', '470', '–9', '–4')
    #   - Numeri romani isolati
    #   - Pattern come 'pNUM' (es. p324)
    #   - Numeri attaccati a "Index" (es. "852Index 868")
    testo_pulito = re.sub(r'\b\d{1,4}(?:\s*-\s*\d{1,4})?\b', ' ', testo_pulito)
    testo_pulito = re.sub(r'\b[Pp]\d{1,4}(?: \d{1,4})?\b', ' ', testo_pulito)
    testo_pulito = re.sub(r'\b[ivxIVX]{1,4}\b', ' ', testo_pulito)
    testo_pulito = re.sub(r'\bIndex\s*\d+\s*', ' ', testo_pulito)
    testo_pulito = re.sub(r'\b\d+th\s*century\s*BC\b', ' ', testo_pulito)  # Per "4th century BC"

    # 9. Rimozione simboli speciali e caratteri non alfanumerici (whitelist)
    # Lascia lettere, numeri, spazi e punteggiatura essenziale per il testo (.,!?;:'"-)
    # Rimuove anche simboli come , , , , , °, 
    testo_pulito = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\:\;\'\"\(\)\-\–—]', ' ', testo_pulito)

    # 10. Correzioni specifiche di parole unite/spezzate da OCR e ligatures residue
    testo_pulito = testo_pulito.replace('clinicalmedicine', 'clinical medicine')
    testo_pulito = testo_pulito.replace('defi brillation', 'defibrillation')
    testo_pulito = testo_pulito.replace('fi brillation', 'fibrillation')
    testo_pulito = testo_pulito.replace('confi dentiality', 'confidentiality')
    testo_pulito = testo_pulito.replace('suff erers', 'sufferers')
    testo_pulito = testo_pulito.replace('refl ect', 'reflect')
    testo_pulito = testo_pulito.replace('eff ort', 'effort')
    testo_pulito = testo_pulito.replace('unaff ordable', 'unaffordable')
    testo_pulito = testo_pulito.replace('hy-giene', 'hygiene')
    testo_pulito = testo_pulito.replace('fi rst', 'first')
    testo_pulito = testo_pulito.replace('pre- cept', 'precept')
    testo_pulito = testo_pulito.replace('offi cer', 'officer')
    testo_pulito = testo_pulito.replace('con- sciousness', 'consciousness')
    testo_pulito = testo_pulito.replace('stetho scope', 'stethoscope')
    testo_pulito = testo_pulito.replace('diff erent', 'different')
    testo_pulito = testo_pulito.replace('ground- breaking', 'ground-breaking')
    testo_pulito = testo_pulito.replace('prac- tice-changing', 'practice-changing')
    testo_pulito = testo_pulito.replace('aff ordable', 'affordable')
    testo_pulito = testo_pulito.replace('refl ection', 'reflection')
    testo_pulito = testo_pulito.replace('con- sciousness', 'consciousness')
    testo_pulito = testo_pulito.replace('catego- ries', 'categories')
    testo_pulito = testo_pulito.replace('uti litarian', 'utilitarian')
    testo_pulito = testo_pulito.replace('indi vidual', 'individual')
    testo_pulito = testo_pulito.replace('re- sponsibility', 'responsibility')
    testo_pulito = testo_pulito.replace('refl ect', 'reflect')
    testo_pulito = testo_pulito.replace('suff erers', 'sufferers')
    testo_pulito = testo_pulito.replace('admon ished', 'admonished')
    testo_pulito = testo_pulito.replace('trans - gress', 'transgress')
    testo_pulito = testo_pulito.replace('con - fi dentiality', 'confidentiality')
    testo_pulito = testo_pulito.replace('re - gard', 'regard')
    testo_pulito = testo_pulito.replace('bene - fi t', 'benefit')
    testo_pulito = testo_pulito.replace('con - sciousness', 'consciousness')
    testo_pulito = testo_pulito.replace('professi on', 'profession')
    testo_pulito = testo_pulito.replace('con - jecture', 'conjecture')
    testo_pulito = testo_pulito.replace('eff ort', 'effort')
    testo_pulito = testo_pulito.replace('con - viction', 'conviction')
    testo_pulito = testo_pulito.replace('aporia of Socrates: At fi rst', 'aporia of Socrates: At first')
    testo_pulito = testo_pulito.replace('non-verbal cut-off s', 'non-verbal cut-offs')
    testo_pulito = testo_pulito.replace('dis- torted', 'distorted')
    testo_pulito = testo_pulito.replace('stetho scope', 'stethoscope')
    testo_pulito = testo_pulito.replace('fi nancing', 'financing')
    testo_pulito = testo_pulito.replace('effi ciency', 'efficiency')
    testo_pulito = testo_pulito.replace('eff ective', 'effective')
    testo_pulito = testo_pulito.replace('off ering', 'offering')
    testo_pulito = testo_pulito.replace('specifi c', 'specific')
    testo_pulito = testo_pulito.replace('Teach- ing', 'Teaching')
    testo_pulito = testo_pulito.replace('tire- less', 'tireless')
    testo_pulito = testo_pulito.replace('stu- dents', 'students')
    testo_pulito = testo_pulito.replace('edi- tions', 'editions')
    testo_pulito = testo_pulito.replace('Heart- felt', 'Heart-felt')
    testo_pulito = testo_pulito.replace('off - ering', 'offering')
    testo_pulito = testo_pulito.replace('pre - cept', 'precept')
    testo_pulito = testo_pulito.replace('dei-fi ed', 'deified')  # Nuovo da input
    testo_pulito = testo_pulito.replace('Gal- lery', 'Gallery')  # Nuovo da input
    testo_pulito = testo_pulito.replace('Mansfi eld', 'Mansfield')  # Nuovo da input
    testo_pulito = testo_pulito.replace('fulfi l', 'fulfill')  # Nuovo da input
    testo_pulito = testo_pulito.replace('judgement', 'judgment')  # Nuovo da input
    testo_pulito = testo_pulito.replace('off spring', 'offspring')  # Nuovo da input
    testo_pulito = testo_pulito.replace('bene fi t', 'benefit')  # Nuovo da input
    testo_pulito = testo_pulito.replace('pro- cure', 'procure')  # Nuovo da input
    testo_pulito = testo_pulito.replace('Whatsoever things', 'Whatsoever things')  # Nuovo da input (già ok)
    testo_pulito = testo_pulito.replace('confi ned', 'confined')  # Nuovo da input
    testo_pulito = testo_pulito.replace('scientifi c', 'scientific')  # Nuovo da input
    testo_pulito = testo_pulito.replace('con- sciousness', 'consciousness')  # Nuovo da input
    testo_pulito = testo_pulito.replace('aporia of Socrates: At fi rst',
                                        'aporia of Socrates: At first')  # Nuovo da input
    testo_pulito = testo_pulito.replace('scien tifi c', 'scientific')  # Nuovo da input
    testo_pulito = testo_pulito.replace('fi nite', 'finite')  # Nuovo da input
    testo_pulito = testo_pulito.replace('con- sciousness', 'consciousness')  # Nuovo da input
    testo_pulito = testo_pulito.replace('fi ght', 'fight')  # Nuovo da input
    testo_pulito = testo_pulito.replace('off er', 'offer')  # Nuovo da input
    testo_pulito = testo_pulito.replace('const ellation', 'constellation')  # Nuovo da input
    testo_pulito = testo_pulito.replace('Bel- latrix', 'Bellatrix')  # Nuovo da input
    testo_pulito = testo_pulito.replace('Betel geuse', 'Betelgeuse')  # Nuovo da input
    testo_pulito = testo_pulito.replace('Mint a ka', 'Mintaka')  # Nuovo da input
    testo_pulito = testo_pulito.replace('effi cacy', 'efficacy')  # Nuovo da input
    testo_pulito = testo_pulito.replace('benefi t', 'benefit')  # Nuovo da input
    testo_pulito = testo_pulito.replace('diff erent', 'different')  # Nuovo da input
    testo_pulito = testo_pulito.replace('fi ctional', 'fictional')  # Nuovo da input
    testo_pulito = testo_pulito.replace('Overconfi dence', 'Overconfidence')  # Nuovo da input
    testo_pulito = testo_pulito.replace('con- fi ned', 'confined')  # Nuovo da input

    # 11. Pulizia di spazi multipli e ritaglio finale
    testo_pulito = re.sub(r'\s+', ' ', testo_pulito).strip()

    return testo_pulito

# --------------------------------------------------------------------------------------
# FUNZIONE PRINCIPALE DI CONVERSIONE AGGIORNATA
# --------------------------------------------------------------------------------------

def converti_pdf_in_txt(input_pdf_path_, output_txt_path_):
    """
    Estrae il testo da un file PDF, lo pulisce e lo salva in un file .txt.
    """
    print(f"Inizio estrazione da: {input_pdf_path_}")
    testo_estratto = ""
    try:
        output_string = StringIO()

        with open(input_pdf_path_, 'rb') as input_pdf_file:
            # extract_text_to_fp è la fase di estrazione
            extract_text_to_fp(input_pdf_file, output_string)

        testo_estratto = output_string.getvalue()

        # >>> FASE DI PULIZIA CRITICA <<<
        print("Eseguo la pulizia del testo...")
        testo_pulito = pulisci_testo_per_pretraining(testo_estratto)

        # Scrive il testo pulito nel file .txt
        with open(output_txt_path_, 'w', encoding='utf-8') as output_txt_file:
            output_txt_file.write(testo_pulito)

        print(f"✔ Conversione e pulizia completate. Salvato in: '{output_txt_path_}'")

    except FileNotFoundError:
        print(f"❌ Errore: File PDF non trovato al percorso: {input_pdf_path_}")
    except Exception as e:
        print(f"❌ Si è verificato un errore durante la conversione/pulizia: {e}")


# --------------------------------------------------------------------------------------
# ESECUZIONE
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("__main__() - BEGIN")
    pdf_document_path = "../../../Datasets/Pdf_Books/Oxford_Handbook_of_Clinical_Medicine_10th_2017_Edition_SamanSarKo.pdf"
    text_document_path = "../../../Datasets/Txt_Books/Oxford_Handbook_of_Clinical_Medicine_10th_2017_Edition_SamanSarKo.txt"

    converti_pdf_in_txt(pdf_document_path, text_document_path)

    print("__main__() - END")