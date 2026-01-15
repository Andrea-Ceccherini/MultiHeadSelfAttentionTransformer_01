from transformers import AutoTokenizer


def load_gpt2_tokenizer():
    print("\nload_gpt2_tokenizer() - BEGIN")
    tokenizer_ = None
    tokenizer_len_ = None

    try:
        # tokenizer_ = AutoTokenizer.from_pretrained("gpt2")
        # tokenizer_ = AutoTokenizer.from_pretrained("gpt2-medium")
        # tokenizer_ = AutoTokenizer.from_pretrained("gpt2-large")
        # tokenizer_ = AutoTokenizer.from_pretrained("gpt2-xl")
        # tokenizer_ = AutoTokenizer.from_pretrained("distilgpt2")
        # tokenizer_ = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        # tokenizer_ = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf") # Requires a Hugging Face User Access Token
        # tokenizer_ = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        # tokenizer_ = AutoTokenizer.from_pretrained("t5-base")
        # tokenizer_ = AutoTokenizer.from_pretrained("facebook/bart-large")
        # tokenizer_ = AutoTokenizer.from_pretrained("bert-base-uncased")
        # tokenizer_ = AutoTokenizer.from_pretrained("roberta-base")
        tokenizer_ = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")



        if tokenizer_.pad_token is None:
            tokenizer_.add_special_tokens({'pad_token': '[PAD]'})

        tokenizer_len_ = len(tokenizer_)
    except Exception as e:
        print(f"load_gpt2_tokenizer() - Error loading tokenizer_: {e}")

    print("load_gpt2_tokenizer() - END\n")
    return tokenizer_, tokenizer_len_



if __name__ == "__main__":
    print("__main__ - BEGIN")
    tokenizer, tokenizer_len = load_gpt2_tokenizer()
    tokenizer_sizeof = tokenizer.__sizeof__()
    print("__main__() - tokenizer =", tokenizer)
    print("__main__() - tokenizer_len =", tokenizer_len)
    print("__main__() - tokenizer_sizeof =", tokenizer_sizeof)

    print("__main__ - END")
