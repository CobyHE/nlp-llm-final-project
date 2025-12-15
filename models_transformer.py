# models_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """RMSNorm: x / rms(x) * weight (no mean subtraction). No bias."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def make_norm(norm_type: str, d_model: int):
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)
    elif norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x):
        # x is only used for shape/device; return (1, T, D)
        # x: (B, T, D)
        T = x.size(1)
        return self.pe[:T].unsqueeze(0)  # (1, T, D)


def _causal_mask_bool(T: int, device):
    # True means masked-out positions (upper triangle above diagonal)
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class EncoderLayer(nn.Module):
    """Pre-norm Transformer Encoder layer (batch_first)."""
    def __init__(self, d_model, nhead, dim_ff, dropout, norm_type="layernorm"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, S, D)
        # src_key_padding_mask: (B, S) True for PAD
        h = self.norm1(x)
        attn_out, _ = self.self_attn(
            h, h, h,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class DecoderLayer(nn.Module):
    """Pre-norm Transformer Decoder layer (batch_first)."""
    def __init__(self, d_model, nhead, dim_ff, dropout, norm_type="layernorm"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)
        self.norm3 = make_norm(norm_type, d_model)

    def forward(
        self,
        x,                      # (B, T, D)
        memory,                 # (B, S, D)
        tgt_mask=None,          # (T, T) bool, True = masked
        tgt_key_padding_mask=None,      # (B, T) bool, True = PAD
        memory_key_padding_mask=None,   # (B, S) bool, True = PAD
    ):
        # 1) masked self-attn
        h = self.norm1(x)
        self_out, _ = self.self_attn(
            h, h, h,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(self_out)

        # 2) cross-attn
        h = self.norm2(x)
        cross_out, _ = self.cross_attn(
            h, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(cross_out)

        # 3) ffn
        h = self.norm3(x)
        x = x + self.dropout(self.ffn(h))
        return x


class TransformerNMT(nn.Module):
    """
    Encoder-Decoder Transformer for NMT
    - pos_type: learned | sinusoidal
    - norm_type: layernorm | rmsnorm
    """
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=512,
        nhead=8,
        num_enc_layers=4,
        num_dec_layers=4,
        dim_ff=2048,
        dropout=0.1,
        max_len=256,
        pos_type="learned",
        norm_type="layernorm",
        pad_id=0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.pos_type = pos_type
        self.norm_type = norm_type

        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)

        # absolute position embeddings
        if pos_type == "learned":
            self.src_pos = nn.Embedding(max_len, d_model)
            self.tgt_pos = nn.Embedding(max_len, d_model)
        elif pos_type == "sinusoidal":
            self.src_pos = SinusoidalPositionalEmbedding(max_len, d_model)
            self.tgt_pos = SinusoidalPositionalEmbedding(max_len, d_model)
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_ff, dropout, norm_type=norm_type)
            for _ in range(num_enc_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_ff, dropout, norm_type=norm_type)
            for _ in range(num_dec_layers)
        ])

        # final projection
        self.out = nn.Linear(d_model, tgt_vocab)

    def _pos(self, B, T, device, which="src"):
        if self.pos_type == "learned":
            idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            return self.src_pos(idx) if which == "src" else self.tgt_pos(idx)  # (B,T,D)
        else:
            dummy = torch.empty(B, T, self.d_model, device=device)
            return self.src_pos(dummy) if which == "src" else self.tgt_pos(dummy)  # (1,T,D)

    def make_pad_mask(self, x):
        # x: (B,T) -> True where PAD
        return (x == self.pad_id)

    def forward(self, src, tgt_in):
        """
        src: (B,S)
        tgt_in: (B,T) decoder input starts with <SOS>
        returns logits: (B,T,V)
        """
        B, S = src.size()
        B2, T = tgt_in.size()
        assert B == B2

        src_pad = self.make_pad_mask(src)      # (B,S) bool
        tgt_pad = self.make_pad_mask(tgt_in)   # (B,T) bool
        tgt_mask = _causal_mask_bool(T, src.device)  # (T,T) bool

        # embeddings + positions
        src_x = self.src_emb(src) * math.sqrt(self.d_model)
        tgt_x = self.tgt_emb(tgt_in) * math.sqrt(self.d_model)

        src_x = src_x + self._pos(B, S, src.device, "src")
        tgt_x = tgt_x + self._pos(B, T, src.device, "tgt")

        # encoder
        memory = src_x
        for layer in self.enc_layers:
            memory = layer(memory, src_key_padding_mask=src_pad)

        # decoder
        out = tgt_x
        for layer in self.dec_layers:
            out = layer(
                out,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad,
                memory_key_padding_mask=src_pad,
            )

        logits = self.out(out)
        return logits
