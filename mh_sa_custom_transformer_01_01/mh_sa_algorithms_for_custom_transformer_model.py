from transformers import PreTrainedTokenizerFast
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

NUM_LAYERS = 12
D_MODEL = 768
TOKENIZATION_MAX_LENGTH = 256   # To prevent answer cut off. Gives the model more "runway" allows it to read longer Wikipedia contexts in Phase 1, making it smarter at constructing sentences.
NUM_HEADS = 12
D_FF = 3072
DROPOUT = 0.2   # In Transformer theory, the Feed Forward layer is usually 4 times the size of the Model Dimension (768 * 4). This gives the model more capacity to store "facts".


class CustomTransformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder_block=False) for _ in range(num_layers)])

        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder_block=True) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # --- Encoder ---
        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)

        for layer in self.encoder_layers:
            src = layer(src, self_attn_mask=src_mask)

        encoder_output = src

        # --- Decoder ---
        trg = self.decoder_embedding(trg)
        trg = self.positional_encoding(trg)

        # Generate Causal Mask for Decoder
        trg_seq_len = trg.size(1)

        # --- FIX: Pass dtype=trg.dtype to match FP16/FP32 ---
        causal_mask = generate_square_subsequent_mask(trg_seq_len, trg.device, dtype=trg.dtype)

        if trg_mask is not None:
            causal_mask = causal_mask + trg_mask

        for layer in self.decoder_layers:
            trg = layer(trg, encoder_output=encoder_output, self_attn_mask=causal_mask, cross_attn_mask=src_mask)

        output = self.fc_out(trg)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))



def load_liver_tokenizer():
    print("\nload_liver_tokenizer() - BEGIN")
    tokenizer_path_ = "custom_tokenizer_files"
    tokenizer_ = None
    tokenizer_len = None

    try:
        # Load from local folder

        tokenizer_ = PreTrainedTokenizerFast.from_pretrained(tokenizer_path_)
        print("load_liver_tokenizer() - Local tokenizer found")

        # Ensure special tokens are set
        if tokenizer_.pad_token is None:
            tokenizer_.pad_token = tokenizer_.eos_token

        tokenizer_len = len(tokenizer_)
        # return tokenizer_, len(tokenizer_)

    except Exception:
        # Fallback if local not found
        print("load_liver_tokenizer() - Custom tokenizer not found, falling back to standard GPT-2")
        from transformers import AutoTokenizer
        tokenizer_ = AutoTokenizer.from_pretrained("gpt2")
        tokenizer_.pad_token = tokenizer_.eos_token
        tokenizer_len = len(tokenizer_)

        # return tokenizer_, len(tokenizer_)

    print("load_liver_tokenizer() - END\n")

    return tokenizer_, tokenizer_len


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, is_decoder_block=False):
        super().__init__()
        self.is_decoder_block = is_decoder_block

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        if self.is_decoder_block:
            self.cross_attention = MultiHeadAttention(d_model, num_heads)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None):
        attn_output = self.dropout1(self.self_attention(x, x, x, self_attn_mask))
        x = self.norm1(x + attn_output)

        if self.is_decoder_block and encoder_output is not None:
            cross_attn_output = self.dropout2(self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask))
            x = self.norm2(x + cross_attn_output)

        ff_output = self.dropout3(self.feed_forward(x))
        x = self.norm3(x + ff_output)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        query = split_heads(self.query(query))
        key = split_heads(self.key(key))
        value = split_heads(self.value(value))

        attention_output, _ = scaled_dot_product_attention(query, key, value, mask)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(attention_output)




def generate_square_subsequent_mask(sz, device, dtype=torch.float32):
    """Generates a causal mask (upper triangular -inf) with specific dtype."""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    # Convert to the correct dtype (FP16 or FP32) BEFORE filling
    mask = mask.to(dtype).masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    # FIX: Ensure the scaling factor matches the query's data type (FP16 or FP32)
    scale = torch.sqrt(torch.tensor(d_k, device=query.device, dtype=query.dtype))

    scores = torch.matmul(query, key.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores + mask

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights