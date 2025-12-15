#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import argparse
import random
from typing import List, Optional

import torch

# project imports
from dataset import load_vocab, SPECIALS
from models_rnn import Encoder, Decoder, sequence_mask
from models_transformer import TransformerNMT

# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device(device_str: str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def write_lines(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def safe_print(s: str):
    print(s, flush=True)

def maybe_import_jieba():
    try:
        import jieba  # type: ignore
        return jieba
    except Exception:
        return None

def tokenize_zh(text: str) -> List[str]:
    """
    Try best-effort tokenization for your jieba-based vocab.
    Priority:
      1) if text already contains whitespace -> split by whitespace
      2) else if jieba exists -> jieba.cut
      3) else fallback: per-char
    """
    t = text.strip()
    if not t:
        return []
    if any(ch.isspace() for ch in t):
        return [x for x in t.split() if x]

    jieba = maybe_import_jieba()
    if jieba is not None:
        return [x for x in jieba.cut(t) if x.strip()]

    # fallback: char-level
    return list(t)

def ids_to_en_text(ids: List[int], en_itos: List[str]) -> str:
    toks = []
    eos = SPECIALS["<EOS>"]
    for i in ids:
        if i == eos:
            break
        toks.append(en_itos[i])
    return " ".join(toks)

def zh_text_to_ids(text: str, zh_stoi: dict) -> List[int]:
    unk = SPECIALS.get("<UNK>", None)
    if unk is None:
        # very rare, but keep safe
        unk = SPECIALS["<PAD>"]
    toks = tokenize_zh(text)
    return [zh_stoi.get(tok, unk) for tok in toks]

# -------------------------
# RNN decoding
# -------------------------
@torch.no_grad()
def rnn_decode_greedy(
    encoder: Encoder,
    decoder: Decoder,
    src_ids: torch.Tensor,
    src_len: torch.Tensor,
    max_len: int,
    sos_id: int,
    eos_id: int,
) -> List[int]:
    enc_out, h = encoder(src_ids, src_len)
    mask = sequence_mask(src_len, enc_out.size(1))
    B = src_ids.size(0)
    y_t = torch.full((B,), sos_id, dtype=torch.long, device=src_ids.device)

    hy = tuple(h) if isinstance(h, tuple) else h
    outs = []
    for _ in range(max_len):
        logits, hy, _ = decoder.forward_step(y_t, hy, enc_out, mask)
        y_t = logits.argmax(dim=-1)
        outs.append(y_t)
    outs = torch.stack(outs, dim=1)  # (B,T)

    # B==1 in inference here
    seq = []
    for tok in outs[0].tolist():
        if tok == eos_id:
            break
        seq.append(tok)
    return seq

@torch.no_grad()
def rnn_beam_search(
    encoder: Encoder,
    decoder: Decoder,
    src_ids: torch.Tensor,
    src_len: torch.Tensor,
    max_len: int,
    sos_id: int,
    eos_id: int,
    beam: int = 4,
) -> List[int]:
    # batch=1
    assert src_ids.size(0) == 1
    enc_out, h = encoder(src_ids, src_len)
    mask = sequence_mask(src_len, enc_out.size(1))

    beams = [(0.0, [sos_id], h)]  # (logprob, seq, hidden)
    finished = []

    for _ in range(max_len):
        new_beams = []
        for logp, seq, hid in beams:
            y_t = torch.tensor([seq[-1]], device=src_ids.device)
            logits, hid2, _ = decoder.forward_step(y_t, hid, enc_out, mask)
            logprobs = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk = torch.topk(logprobs, beam)

            for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                new_seq = seq + [tok]
                new_logp = logp + lp
                if tok == eos_id:
                    finished.append((new_logp, new_seq))
                else:
                    new_beams.append((new_logp, new_seq, hid2))

        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam]
        if not beams:
            break

    if not finished:
        finished = [(lp, seq) for lp, seq, _ in beams]
    finished.sort(key=lambda x: x[0], reverse=True)
    best = finished[0][1]
    # drop SOS
    return best[1:]

# -------------------------
# Transformer decoding
# -------------------------
@torch.no_grad()
def tfm_greedy_decode_ids(model: TransformerNMT, src: torch.Tensor, max_len: int, sos_id: int, eos_id: int) -> List[int]:
    model.eval()
    y = torch.full((1, 1), sos_id, dtype=torch.long, device=src.device)
    for _ in range(max_len):
        logits = model(src, y)  # (1,T,V)
        nxt = logits[:, -1].argmax(-1)  # (1,)
        y = torch.cat([y, nxt.unsqueeze(1)], dim=1)
        if nxt.item() == eos_id:
            break
    out = []
    for t in y[0, 1:].tolist():  # drop SOS
        if t == eos_id:
            break
        out.append(t)
    return out

@torch.no_grad()
def tfm_beam_search_single(model: TransformerNMT, src: torch.Tensor, beam: int, max_len: int, sos_id: int, eos_id: int) -> List[int]:
    device = src.device
    model.eval()
    beams = [(0.0, torch.tensor([[sos_id]], device=device, dtype=torch.long))]

    for _ in range(max_len):
        new_beams = []
        for logp, y in beams:
            logits = model(src, y)
            logprobs = torch.log_softmax(logits[:, -1], dim=-1).squeeze(0)
            topk = torch.topk(logprobs, beam)
            for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                y2 = torch.cat([y, torch.tensor([[tok]], device=device, dtype=torch.long)], dim=1)
                new_beams.append((logp + lp, y2))

        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam]

        if all(b[1][0, -1].item() == eos_id for b in beams):
            break

    best = beams[0][1][0].tolist()[1:]  # drop SOS
    out = []
    for t in best:
        if t == eos_id:
            break
        out.append(t)
    return out

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["rnn", "transformer", "mt5"], required=True)

    ap.add_argument("--text", type=str, default=None, help="single input sentence (Chinese)")
    ap.add_argument("--input_file", type=str, default=None, help="batch input file, one sentence per line")
    ap.add_argument("--output_file", type=str, default=None, help="where to write outputs (for batch). If not set, print to stdout")

    # for rnn/transformer
    ap.add_argument("--vocab_zh", type=str, default=None)
    ap.add_argument("--vocab_en", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)

    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=4)
    ap.add_argument("--max_gen", type=int, default=80)

    # for mt5
    ap.add_argument("--mt5_dir", type=str, default=None)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    # input
    if (args.text is None) == (args.input_file is None):
        raise ValueError("Provide exactly one of: --text OR --input_file")

    inputs = [args.text] if args.text is not None else read_lines(args.input_file)

    # -------------------------
    # mt5 inference
    # -------------------------
    if args.model_type == "mt5":
        if args.mt5_dir is None:
            raise ValueError("--mt5_dir is required when --model_type mt5")

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(args.mt5_dir, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.mt5_dir, local_files_only=True).to(device)
        model.eval()

        # You can tweak prompt style if you want:
        # Many T5/mt5 checkpoints work better with a task prefix.
        # Here we keep it minimal and robust:
        def build_prompt(zh: str) -> str:
            return f"translate Chinese to English: {zh.strip()}"

        outs = []
        for s in inputs:
            prompt = build_prompt(s)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            gen = model.generate(
                **enc,
                max_length=args.max_gen,
                num_beams=(args.beam_size if args.decode == "beam" else 1),
                do_sample=False,
            )
            txt = tokenizer.decode(gen[0], skip_special_tokens=True)
            outs.append(txt)

        if args.output_file:
            write_lines(args.output_file, outs)
        else:
            for o in outs:
                safe_print(o)
        return

    # -------------------------
    # rnn/transformer inference need vocab + ckpt
    # -------------------------
    if args.vocab_zh is None or args.vocab_en is None:
        raise ValueError("--vocab_zh and --vocab_en are required for rnn/transformer")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for rnn/transformer")

    zh_stoi, zh_itos = load_vocab(args.vocab_zh)
    en_stoi, en_itos = load_vocab(args.vocab_en)

    sos = SPECIALS["<SOS>"]
    eos = SPECIALS["<EOS>"]
    pad = SPECIALS["<PAD>"]

    ckpt = torch.load(args.checkpoint, map_location=device)

    outputs = []

    if args.model_type == "rnn":
        if "encoder" not in ckpt or "decoder" not in ckpt:
            raise ValueError(
                "This checkpoint does not look like an RNN checkpoint. "
                "Expected keys: 'encoder' and 'decoder'. "
                "Did you pass a transformer checkpoint by mistake?"
            )

        # infer config (or fallback)
        cfg = ckpt.get("config", {})
        emb = int(cfg.get("emb", 256))
        hidden = int(cfg.get("hidden", 512))
        rnn_type = cfg.get("rnn_type", "lstm")
        attn = cfg.get("attn", "dot")

        encoder = Encoder(len(zh_itos), emb, hidden, num_layers=2, rnn_type=rnn_type).to(device)
        decoder = Decoder(len(en_itos), emb, hidden, num_layers=2, rnn_type=rnn_type, attn_type=attn).to(device)

        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        encoder.eval()
        decoder.eval()

        for s in inputs:
            src_list = zh_text_to_ids(s, zh_stoi)
            if len(src_list) == 0:
                outputs.append("")
                continue

            src = torch.tensor([src_list], dtype=torch.long, device=device)
            src_len = torch.tensor([len(src_list)], dtype=torch.long, device=device)

            if args.decode == "greedy":
                pred = rnn_decode_greedy(encoder, decoder, src, src_len, args.max_gen, sos, eos)
            else:
                pred = rnn_beam_search(encoder, decoder, src, src_len, args.max_gen, sos, eos, beam=args.beam_size)

            outputs.append(ids_to_en_text(pred, en_itos))

    elif args.model_type == "transformer":
        if "model" not in ckpt:
            raise ValueError(
                "This checkpoint does not look like a Transformer checkpoint. "
                "Expected key: 'model'. "
                "Did you pass an RNN checkpoint by mistake?"
            )

        cfg = ckpt.get("config", {})
        d_model = int(cfg.get("d_model", 256))
        nhead = int(cfg.get("nhead", 4))
        enc_layers = int(cfg.get("enc_layers", 3))
        dec_layers = int(cfg.get("dec_layers", 3))
        ff = int(cfg.get("ff", 1024))
        dropout = float(cfg.get("dropout", 0.1))
        max_len = int(cfg.get("max_len", 256))
        pos = cfg.get("pos", "learned")
        norm = cfg.get("norm", "layernorm")

        model = TransformerNMT(
            src_vocab=len(zh_itos),
            tgt_vocab=len(en_itos),
            d_model=d_model,
            nhead=nhead,
            num_enc_layers=enc_layers,
            num_dec_layers=dec_layers,
            dim_ff=ff,
            dropout=dropout,
            max_len=max_len,
            pos_type=pos,
            norm_type=norm,
            pad_id=pad,
        ).to(device)

        model.load_state_dict(ckpt["model"])
        model.eval()

        for s in inputs:
            src_list = zh_text_to_ids(s, zh_stoi)
            if len(src_list) == 0:
                outputs.append("")
                continue
            src = torch.tensor([src_list], dtype=torch.long, device=device)

            if args.decode == "greedy":
                pred = tfm_greedy_decode_ids(model, src, args.max_gen, sos, eos)
            else:
                pred = tfm_beam_search_single(model, src, args.beam_size, args.max_gen, sos, eos)

            outputs.append(ids_to_en_text(pred, en_itos))

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if args.output_file:
        write_lines(args.output_file, outputs)
    else:
        for o in outputs:
            safe_print(o)

if __name__ == "__main__":
    main()
