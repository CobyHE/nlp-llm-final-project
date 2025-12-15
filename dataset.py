# dataset.py
import json, torch
from torch.utils.data import Dataset, DataLoader

SPECIALS = {"<PAD>":0, "<UNK>":1, "<SOS>":2, "<EOS>":3}

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    stoi = obj["stoi"]; itos = obj["itos"]
    return stoi, itos

class NMTJsonlDataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                self.rows.append(json.loads(line))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "src_ids": torch.tensor(r["src_ids"], dtype=torch.long),
            "tgt_ids": torch.tensor(r["tgt_ids"], dtype=torch.long),
        }

def pad_sequence_1d(seqs, pad_id):
    maxlen = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=seqs[0].dtype)
    lens = []
    for i, s in enumerate(seqs):
        out[i, :s.size(0)] = s
        lens.append(s.size(0))
    return out, torch.tensor(lens, dtype=torch.long)

def collate_fn(batch, pad_id=0):
    src_ids = [b["src_ids"] for b in batch]
    tgt_ids = [b["tgt_ids"] for b in batch]
    src_pad, src_len = pad_sequence_1d(src_ids, pad_id)
    tgt_pad, tgt_len = pad_sequence_1d(tgt_ids, pad_id)
    # decoder 输入/标签：y_in = 去掉最后一位，y_out = 去掉第一位
    y_in  = tgt_pad[:, :-1]
    y_out = tgt_pad[:, 1:]
    return {
        "src": src_pad, "src_len": src_len,
        "y_in": y_in, "y_out": y_out, "y_len": tgt_len - 1
    }

def make_loader(path, batch_size, shuffle):
    ds = NMTJsonlDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: collate_fn(b, pad_id=SPECIALS["<PAD>"]))
