import json, os, re, argparse
from collections import Counter

SPECIALS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

def clean_text(s: str) -> str:
    """基础清洗：去除非法字符、合并多空格、去掉零宽空格"""
    s = re.sub(r"[^\S\r\n]+", " ", s)
    s = s.replace("\u200b", "")
    s = s.strip()
    return s

def try_import_jieba():
    try:
        import jieba
        return jieba
    except Exception:
        return None

def zh_tokenize(s: str, mode="auto"):
    s = clean_text(s)
    if mode == "char":
        return list(s)
    if mode == "jieba":
        jb = try_import_jieba()
        if jb is None:
            return list(s)
        return [tok for tok in jb.cut(s) if tok.strip()]
    # default = auto
    jb = try_import_jieba()
    if jb is None:
        return list(s)
    return [tok for tok in jb.cut(s) if tok.strip()]

def en_tokenize(s: str):
    s = clean_text(s)
    # 空格+标点分词
    tokens = []
    for w in s.split():
        w = re.sub(r"([.,!?;:()\"'])", r" \1 ", w)
        tokens.extend([t for t in w.split() if t])
    return tokens

def build_vocab(token_lists, min_freq: int, max_size: int=None):
    cnt = Counter()
    for toks in token_lists:
        cnt.update(toks)
    items = [(tok, f) for tok, f in cnt.items() if f >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_size is not None:
        items = items[:max(0, max_size - len(SPECIALS))]
    itos = SPECIALS + [tok for tok, _ in items]
    stoi = {tok:i for i, tok in enumerate(itos)}
    return stoi, itos, cnt

def encode(tokens, stoi, max_len):
    ids = [stoi.get("<SOS>")]
    for t in tokens:
        ids.append(stoi.get(t, stoi["<UNK>"]))
    ids.append(stoi.get("<EOS>"))
    if max_len is not None:
        ids = ids[:max_len]
    return ids

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            obj = json.loads(line)
            if "zh" in obj and "en" in obj:
                yield obj["zh"], obj["en"]
            elif "chinese" in obj and "english" in obj:
                yield obj["chinese"], obj["english"]
            else:
                raise ValueError(f"Unknown json schema in {path}: {obj.keys()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_freq", type=int, default=3)
    ap.add_argument("--max_src_len", type=int, default=60)
    ap.add_argument("--max_tgt_len", type=int, default=60)
    ap.add_argument("--max_vocab", type=int, default=50000)
    ap.add_argument("--zh_tokenizer", choices=["auto","jieba","char"], default="auto")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Step 1. 加载训练数据并分词
    zh_train_tokens, en_train_tokens, train_pairs = [], [], []
    for zh, en in load_jsonl(args.train):
        zh_toks = zh_tokenize(zh, mode=args.zh_tokenizer)
        en_toks = en_tokenize(en)
        if len(zh_toks) == 0 or len(en_toks) == 0:
            continue
        if len(zh_toks) > args.max_src_len or len(en_toks) > args.max_tgt_len:
            continue
        zh_train_tokens.append(zh_toks)
        en_train_tokens.append(en_toks)
        train_pairs.append((zh_toks, en_toks))

    # Step 2. 构建词表
    zh_stoi, zh_itos, _ = build_vocab(zh_train_tokens, args.min_freq, args.max_vocab)
    en_stoi, en_itos, _ = build_vocab(en_train_tokens, args.min_freq, args.max_vocab)

    # 保存词表
    with open(os.path.join(args.outdir, "vocab_zh.json"), "w", encoding="utf-8") as f:
        json.dump({"stoi": zh_stoi, "itos": zh_itos}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "vocab_en.json"), "w", encoding="utf-8") as f:
        json.dump({"stoi": en_stoi, "itos": en_itos}, f, ensure_ascii=False, indent=2)

    # Step 3. 编码 + 保存 tokenized splits
    def process_split(path, prefix):
        data, total, kept = [], 0, 0
        for zh, en in load_jsonl(path):
            total += 1
            zh_toks = zh_tokenize(zh, mode=args.zh_tokenizer)
            en_toks = en_tokenize(en)
            if len(zh_toks) == 0 or len(en_toks) == 0:
                continue
            if len(zh_toks) > args.max_src_len or len(en_toks) > args.max_tgt_len:
                continue
            src_ids = encode(zh_toks, zh_stoi, args.max_src_len+2)
            tgt_ids = encode(en_toks, en_stoi, args.max_tgt_len+2)
            data.append({
                "src_tokens": zh_toks,
                "tgt_tokens": en_toks,
                "src_ids": src_ids,
                "tgt_ids": tgt_ids
            })
            kept += 1
        out_path = os.path.join(args.outdir, f"{prefix}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False)+"\n")
        return total, kept, out_path

    t_total, t_kept, _ = process_split(args.train, "train_tok")
    v_total, v_kept, _ = process_split(args.valid, "valid_tok")
    s_total, s_kept, _ = process_split(args.test, "test_tok")

    stats = {
        "train_total": t_total, "train_kept": t_kept,
        "valid_total": v_total, "valid_kept": v_kept,
        "test_total": s_total, "test_kept": s_kept,
        "zh_vocab_size": len(zh_itos),
        "en_vocab_size": len(en_itos),
        "settings": vars(args)
    }
    with open(os.path.join(args.outdir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
