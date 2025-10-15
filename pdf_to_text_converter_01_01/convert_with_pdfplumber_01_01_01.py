from pdfminer.high_level import extract_text_to_fp
from io import StringIO
import re
import pdfplumber



def pulisci_testo_per_pretraining(testo_grezzo):
    """
    Pulisce il testo estratto dal PDF per renderlo adatto al pre-addestramento di un LLM.
    Questa versione è ancora più aggressiva e include regex più flessibili per i blocchi strutturati.
    """
    testo_pulito = testo_grezzo



    return testo_pulito

# --------------------------------------------------------------------------------------
# FUNZIONE PRINCIPALE DI CONVERSIONE AGGIORNATA
# --------------------------------------------------------------------------------------

def convert_pdf_in_txt(input_pdf_path_, output_txt_path_):
    """
    Estrae il testo da un file PDF, lo pulisce e lo salva in un file .txt.
    """
    print(f"Inizio estrazione da: {input_pdf_path_}")
    testo_estratto = ""
    try:
        with pdfplumber.open(input_pdf_path_) as pdf, open(output_txt_path_, "w", encoding="utf-8") as f:

            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    f.write(t + '\n')


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

    convert_pdf_in_txt(pdf_document_path, text_document_path)

    print("__main__() - END")