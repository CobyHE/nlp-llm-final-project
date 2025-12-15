# models_rnn.py
import torch, torch.nn as nn, torch.nn.functional as F

SPECIALS = {"<PAD>":0, "<UNK>":1, "<SOS>":2, "<EOS>":3}

def sequence_mask(lengths, max_len=None):
    if max_len is None: max_len = lengths.max()
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return arange < lengths.unsqueeze(1)  # (B, T)

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden, num_layers=2, rnn_type="lstm", dropout=0.1, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.rnn_type = rnn_type.lower()
        rnn_cls = nn.LSTM if self.rnn_type=="lstm" else nn.GRU
        self.rnn = rnn_cls(emb_size, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
    def forward(self, x, lengths):
        emb = self.embed(x)                       # (B, S, E)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B,S,H)
        return out, h                             # h: (L,B,H) or tuple for LSTM

class DotAttention(nn.Module):
    def forward(self, q, k, v, mask):  # q: (B,1,H), k/v: (B,S,H)
        # score = q·k
        scores = torch.bmm(q, k.transpose(1,2)).squeeze(1)  # (B,S)
        scores.masked_fill_(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)                    # (B,S)
        ctx = torch.bmm(attn.unsqueeze(1), v).squeeze(1)    # (B,H)
        return ctx, attn

class GeneralAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.Wa = nn.Linear(hidden, hidden, bias=False)
    def forward(self, q, k, v, mask):
        qW = self.Wa(q)                          # (B,1,H)
        scores = torch.bmm(qW, k.transpose(1,2)).squeeze(1)
        scores.masked_fill_(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), v).squeeze(1)
        return ctx, attn

class AdditiveAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.Wq = nn.Linear(hidden, hidden)
        self.Wk = nn.Linear(hidden, hidden)
        self.v  = nn.Linear(hidden, 1, bias=False)
    def forward(self, q, k, v, mask):
        # score = v^T tanh(Wq q + Wk k)
        B, S, H = k.size()
        q_exp = self.Wq(q).expand(-1, S, -1)     # (B,S,H)
        k_lin = self.Wk(k)
        scores = self.v(torch.tanh(q_exp + k_lin)).squeeze(-1)  # (B,S)
        scores.masked_fill_(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), v).squeeze(1)
        return ctx, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden, num_layers=2, rnn_type="lstm",
                 attn_type="dot", dropout=0.1, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.rnn_type = rnn_type.lower()
        rnn_cls = nn.LSTM if self.rnn_type=="lstm" else nn.GRU
        self.rnn = rnn_cls(emb_size, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        if attn_type=="dot":       self.attn = DotAttention()
        elif attn_type=="general": self.attn = GeneralAttention(hidden)
        else:                      self.attn = AdditiveAttention(hidden)  # "additive"
        self.fc = nn.Linear(hidden*2, hidden)
        self.out = nn.Linear(hidden, vocab_size)

    def forward_step(self, y_t, hidden, enc_out, enc_mask):
        # y_t: (B,) next input id
        emb = self.embed(y_t).unsqueeze(1)                 # (B,1,E)
        out, hidden = self.rnn(emb, hidden)                # out: (B,1,H)
        q = out                                            # (B,1,H)
        ctx, attn = self.attn(q, enc_out, enc_out, enc_mask)
        cat = torch.cat([out.squeeze(1), ctx], dim=-1)     # (B,2H)
        h_t = torch.tanh(self.fc(cat))                     # (B,H)
        logits = self.out(h_t)                             # (B,V)
        return logits, hidden, attn

    def forward(self, y_in, hidden, enc_out, enc_mask, teacher_forcing=True):
        # y_in: (B,T) decoder inputs (start with <SOS>)
        B, T = y_in.size()
        logits_list = []
        y_t = y_in[:, 0]                                   # first token (<SOS>)
        for t in range(1, T+1):                            # predict t=1..T
            logit, hidden, _ = self.forward_step(y_t, hidden, enc_out, enc_mask)
            logits_list.append(logit.unsqueeze(1))
            if t < T:
                if teacher_forcing:
                    y_t = y_in[:, t]
                else:
                    y_t = logit.argmax(dim=-1)
        return torch.cat(logits_list, dim=1)               # (B,T, V)
