# eval_transformer_beam.py
import argparse, json, torch
from dataset import make_loader, load_vocab, SPECIALS
from bleu import corpus_bleu
from models_transformer import TransformerNMT

@torch.no_grad()
def beam_search_single(model, src, beam, max_len, sos_id, eos_id):
    """
    src: (1,S)
    returns: best token id sequence (no SOS, cut at EOS)
    """
    device = src.device
    model.eval()
    beams = [(0.0, torch.tensor([[sos_id]], device=device))]  # (logp, y_seq)

    for _ in range(max_len):
        new_beams = []
        for logp, y in beams:
            logits = model(src, y)              # (1, T, V)
            logprobs = torch.log_softmax(logits[:, -1], dim=-1).squeeze(0)
            topk = torch.topk(logprobs, beam)
            for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                y2 = torch.cat([y, torch.tensor([[tok]], device=device)], dim=1)
                new_beams.append((logp + lp, y2))

        # keep top beam
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam]

        # early stop: all beams ended with EOS
        if all(b[1][0, -1].item() == eos_id for b in beams):
            break

    best = beams[0][1][0].tolist()[1:]  # drop SOS
    out = []
    for t in best:
        if t == eos_id:
            break
        out.append(t)
    return out

@torch.no_grad()
def eval_bleu_beam(model, loader, itos_en, device, beam, max_len):
    sos, eos, pad = SPECIALS["<SOS>"], SPECIALS["<EOS>"], SPECIALS["<PAD>"]
    all_cand, all_ref = [], []

    for batch in loader:
        src = batch["src"].to(device)
        y_out = batch["y_out"]  # cpu

        # beam-search per sentence
        cand = []
        for i in range(src.size(0)):
            seq = beam_search_single(
                model, src[i:i+1], beam=beam,
                max_len=max_len, sos_id=sos, eos_id=eos
            )
            cand.append([itos_en[t] for t in seq])

        refs = []
        for row in y_out.tolist():
            toks = []
            for t in row:
                if t == pad:
                    break
                if t == eos:
                    break
                toks.append(itos_en[t])
            refs.append(toks)

        all_cand += cand
        all_ref  += refs

    return corpus_bleu(all_cand, all_ref, max_n=4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--vocab_zh", required=True)
    ap.add_argument("--vocab_en", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load vocab
    _, zh_itos = load_vocab(args.vocab_zh)
    _, en_itos = load_vocab(args.vocab_en)

    # load config & model
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]

    model = TransformerNMT(
        src_vocab=len(zh_itos),
        tgt_vocab=len(en_itos),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_enc_layers=cfg["enc_layers"],
        num_dec_layers=cfg["dec_layers"],
        dim_ff=cfg["ff"],
        dropout=cfg.get("dropout", 0.1),
        max_len=cfg.get("max_len", 256),
        pos_type=cfg["pos"],
        norm_type=cfg["norm"],
        pad_id=SPECIALS["<PAD>"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # loader
    valid_loader = make_loader(
        f"{args.data_dir}/valid_tok.jsonl",
        args.batch_size,
        shuffle=False
    )

    bleu = eval_bleu_beam(
        model, valid_loader, en_itos, device,
        beam=args.beam, max_len=args.max_len
    )
    print(f"Beam={args.beam}  valid BLEU-4 = {bleu*100:.2f}")

if __name__ == "__main__":
    main()
