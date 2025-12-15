# train_transformer.py
import os, json, argparse, random, time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import make_loader, load_vocab, SPECIALS
from bleu import corpus_bleu
from models_transformer import TransformerNMT


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_tag(x) -> str:
    return str(x).replace("/", "_").replace(" ", "").replace(":", "_")

def get_dataset_tag(data_dir: str) -> str:
    return os.path.basename(os.path.normpath(data_dir))

def make_run_name_tfm(args) -> str:
    ds = get_dataset_tag(args.data_dir)
    return safe_tag(
        f"{ds}_tfm_dm{args.d_model}_h{args.nhead}_L{args.enc_layers}-{args.dec_layers}"
        f"_ff{args.ff}_pos-{args.pos}_norm-{args.norm}"
        f"_bs{args.batch_size}_lr{args.lr}_seed{args.seed}"
        f"_trainDec-{args.train_decode}"
    )

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

@torch.no_grad()
def greedy_decode_ids(model, src, max_len, sos_id, eos_id):
    model.eval()
    B = src.size(0)
    y = torch.full((B, 1), sos_id, dtype=torch.long, device=src.device)

    for _ in range(max_len):
        logits = model(src, y)              # (B, T, V)
        next_id = logits[:, -1].argmax(-1)  # (B,)
        y = torch.cat([y, next_id.unsqueeze(1)], dim=1)
        if (next_id == eos_id).all():
            break

    out = []
    for i in range(B):
        seq = []
        for t in y[i, 1:].tolist():  # drop SOS
            if t == eos_id:
                break
            seq.append(t)
        out.append(seq)
    return out


@torch.no_grad()
def beam_search_single(model, src, beam, max_len, sos_id, eos_id):
    device = src.device
    model.eval()
    beams = [(0.0, torch.tensor([[sos_id]], device=device))]

    for _ in range(max_len):
        new_beams = []
        for logp, y in beams:
            logits = model(src, y)
            logprobs = torch.log_softmax(logits[:, -1], dim=-1).squeeze(0)
            topk = torch.topk(logprobs, beam)
            for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                y2 = torch.cat([y, torch.tensor([[tok]], device=device)], dim=1)
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


@torch.no_grad()
def eval_bleu(model, loader, itos_en, device, decode="greedy", beam_size=4, max_gen=80):
    sos, eos, pad = SPECIALS["<SOS>"], SPECIALS["<EOS>"], SPECIALS["<PAD>"]
    all_cand, all_ref = [], []

    for batch in loader:
        src = batch["src"].to(device)
        y_out = batch["y_out"]  # cpu ids for reference

        cand = []
        if decode == "greedy":
            # batch greedy
            pred_ids = greedy_decode_ids(model, src, max_gen, sos, eos)
            for seq in pred_ids:
                cand.append([itos_en[t] for t in seq])
        else:
            # per-sample beam (slow)
            for i in range(src.size(0)):
                seq = beam_search_single(
                    model, src[i:i+1],
                    beam=beam_size,
                    max_len=max_gen,
                    sos_id=sos,
                    eos_id=eos
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
        all_ref += refs

    return corpus_bleu(all_cand, all_ref, max_n=4)


def load_checkpoint(path, model, opt=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt and ckpt["opt"] is not None:
        opt.load_state_dict(ckpt["opt"])
    start_epoch = ckpt.get("epoch", 0)
    best_bleu = ckpt.get("best_bleu", -1.0)
    return start_epoch, best_bleu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--vocab_zh", required=True)
    ap.add_argument("--vocab_en", required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--enc_layers", type=int, default=3)
    ap.add_argument("--dec_layers", type=int, default=3)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=256)

    ap.add_argument("--pos", choices=["learned", "sinusoidal"], default="learned")
    ap.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_root", type=str, default=None)

    # NEW: train decode vs final decode
    ap.add_argument("--beam_size", type=int, default=4)
    ap.add_argument("--train_decode", choices=["greedy", "beam"], default="greedy",
                    help="decode used during training/valid monitoring (recommend greedy)")
    ap.add_argument("--final_decode", choices=["none", "greedy", "beam"], default="beam",
                    help="extra decoding after training for reporting")
    ap.add_argument("--final_beam_size", type=int, default=4)

    # NEW: resume & optional test-each-epoch
    ap.add_argument("--resume", type=str, default=None, help="path to best.pt/last.pt")
    ap.add_argument("--eval_test_each_epoch", action="store_true", help="also eval test each epoch (slow)")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_root = args.save_root or os.path.join(args.data_dir, "checkpoints")
    run_name = make_run_name_tfm(args)
    ckpt_dir = ensure_dir(os.path.join(save_root, "transformer", run_name))

    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    log_path = os.path.join(ckpt_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,valid_bleu4,test_bleu4,seconds\n")

    _, zh_itos = load_vocab(args.vocab_zh)
    _, en_itos = load_vocab(args.vocab_en)

    train_loader = make_loader(f"{args.data_dir}/train_tok.jsonl", args.batch_size, shuffle=True)
    valid_loader = make_loader(f"{args.data_dir}/valid_tok.jsonl", args.batch_size, shuffle=False)
    test_loader  = make_loader(f"{args.data_dir}/test_tok.jsonl",  args.batch_size, shuffle=False)

    model = TransformerNMT(
        src_vocab=len(zh_itos),
        tgt_vocab=len(en_itos),
        d_model=args.d_model,
        nhead=args.nhead,
        num_enc_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        dim_ff=args.ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pos_type=args.pos,
        norm_type=args.norm,
        pad_id=SPECIALS["<PAD>"],
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=SPECIALS["<PAD>"])
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    best_bleu = -1.0
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")

    start_epoch = 0
    if args.resume:
        start_epoch, best_bleu = load_checkpoint(args.resume, model, opt=opt, device=device)
        print(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_bleu={best_bleu:.4f})")

    for ep in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0

        for batch in train_loader:
            src = batch["src"].to(device)
            y_in = batch["y_in"].to(device)
            y_out = batch["y_out"].to(device)

            opt.zero_grad()
            logits = model(src, y_in)  # (B,T,V)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        train_loss = total / max(1, len(train_loader))

        valid_bleu = eval_bleu(
            model, valid_loader, en_itos, device,
            decode=args.train_decode, beam_size=args.beam_size
        )

        test_bleu = float("nan")
        if args.eval_test_each_epoch:
            test_bleu = eval_bleu(
                model, test_loader, en_itos, device,
                decode=args.train_decode, beam_size=args.beam_size
            )

        dt = time.time() - t0
        print(f"[Epoch {ep}] train_loss={train_loss:.4f} valid_BLEU4={valid_bleu*100:.2f}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ep},{train_loss:.6f},{valid_bleu*100:.4f},{test_bleu*100 if test_bleu==test_bleu else 'nan'},{dt:.2f}\n")

        # save last
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "config": vars(args),
            "epoch": ep,
            "valid_bleu4": valid_bleu,
            "best_bleu": best_bleu
        }, last_path)

        # save best
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "config": vars(args),
                "epoch": ep,
                "valid_bleu4": valid_bleu,
                "best_bleu": best_bleu
            }, best_path)

    # -------------------------
    # final reporting: load best and compute extra decode
    # -------------------------
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, opt=None, device=device)

    best_valid_trainDec = eval_bleu(
        model, valid_loader, en_itos, device,
        decode=args.train_decode, beam_size=args.beam_size
    )
    best_test_trainDec = eval_bleu(
        model, test_loader, en_itos, device,
        decode=args.train_decode, beam_size=args.beam_size
    )

    extra = {}
    if args.final_decode != "none":
        extra_valid = eval_bleu(
            model, valid_loader, en_itos, device,
            decode=args.final_decode, beam_size=args.final_beam_size
        )
        extra_test = eval_bleu(
            model, test_loader, en_itos, device,
            decode=args.final_decode, beam_size=args.final_beam_size
        )
        extra = {
            f"best_valid_{args.final_decode}_bleu4": extra_valid * 100,
            f"best_test_{args.final_decode}_bleu4": extra_test * 100,
            "final_decode": args.final_decode,
            "final_beam_size": args.final_beam_size
        }

    summary = {
        "run_name": run_name,
        "ckpt_dir": ckpt_dir,
        "best_path": best_path,
        "last_path": last_path,

        "train_decode": args.train_decode,
        "beam_size": args.beam_size,

        "best_valid_trainDecode_bleu4": best_valid_trainDec * 100,
        "best_test_trainDecode_bleu4": best_test_trainDec * 100,
        **extra
    }

    with open(os.path.join(ckpt_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved run dir:", ckpt_dir)
    print("Best:", best_path)
    print("Last:", last_path)
    print(f"Best(valid) [{args.train_decode}] BLEU4 = {best_valid_trainDec*100:.2f} | Best(test) [{args.train_decode}] BLEU4 = {best_test_trainDec*100:.2f}")
    if args.final_decode != "none":
        print(f"Best(valid) [{args.final_decode}] BLEU4 = {extra['best_valid_'+args.final_decode+'_bleu4']:.2f} | Best(test) [{args.final_decode}] BLEU4 = {extra['best_test_'+args.final_decode+'_bleu4']:.2f}")


if __name__ == "__main__":
    main()