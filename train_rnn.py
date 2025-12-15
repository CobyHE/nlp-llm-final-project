# train_rnn.py
import os, json, argparse, random, time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import make_loader, load_vocab, SPECIALS
from models_rnn import Encoder, Decoder, sequence_mask
from bleu import corpus_bleu

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_tag(x) -> str:
    return str(x).replace("/", "_").replace(" ", "").replace(":", "_")

def get_dataset_tag(data_dir: str) -> str:
    return os.path.basename(os.path.normpath(data_dir))

def make_run_name_rnn(args) -> str:
    tf_flag = "fr" if args.free_running else "tf"
    ds = get_dataset_tag(args.data_dir)
    return safe_tag(
        f"{ds}_rnn-{args.rnn_type}_attn-{args.attn}_{tf_flag}"
        f"_trainDec-{args.train_decode}_bs{args.batch_size}_lr{args.lr}"
        f"_emb{args.emb}_h{args.hidden}_seed{args.seed}"
    )

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def decode_greedy(encoder, decoder, src, src_len, max_len, sos_id, eos_id):
    enc_out, h = encoder(src, src_len)
    mask = sequence_mask(src_len, enc_out.size(1))
    B = src.size(0)
    y_t = torch.full((B,), sos_id, dtype=torch.long, device=src.device)
    hy = tuple(h) if isinstance(h, tuple) else h

    outs = []
    for _ in range(max_len):
        logits, hy, _ = decoder.forward_step(y_t, hy, enc_out, mask)
        y_t = logits.argmax(dim=-1)
        outs.append(y_t)

    outs = torch.stack(outs, dim=1)  # (B, T)

    seqs = []
    for i in range(B):
        seq = []
        for tok in outs[i].tolist():
            if tok == eos_id:
                break
            seq.append(tok)
        seqs.append(seq)
    return seqs


def beam_search(encoder, decoder, src, src_len, max_len, sos_id, eos_id, beam=4):
    # batched=1
    assert src.size(0) == 1
    enc_out, h = encoder(src, src_len)
    mask = sequence_mask(src_len, enc_out.size(1))

    beams = [(0.0, [sos_id], h)]  # (logprob, seq, hidden)
    finished = []

    for _ in range(max_len):
        new_beams = []
        for logp, seq, hid in beams:
            y_t = torch.tensor([seq[-1]], device=src.device)
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
    return finished[0][1][1:]  # drop SOS


def ids_to_tokens(seqs, itos):
    return [[itos[i] for i in seq] for seq in seqs]

def train_one_epoch(encoder, decoder, loader, opt, loss_fn, device, teacher_forcing=True):
    encoder.train()
    decoder.train()
    total = 0.0

    for batch in loader:
        src = batch["src"].to(device)
        src_len = batch["src_len"].to(device)
        y_in = batch["y_in"].to(device)
        y_out = batch["y_out"].to(device)

        opt.zero_grad()
        enc_out, h = encoder(src, src_len)
        mask = sequence_mask(src_len, enc_out.size(1))
        logits = decoder(y_in, h, enc_out, mask, teacher_forcing=teacher_forcing)

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        opt.step()
        total += loss.item()

    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_bleu(encoder, decoder, loader, itos_en, device, decode="greedy", beam_size=4):
    encoder.eval()
    decoder.eval()

    all_cand, all_ref = [], []
    sos, eos, pad = SPECIALS["<SOS>"], SPECIALS["<EOS>"], SPECIALS["<PAD>"]

    for batch in loader:
        src = batch["src"].to(device)
        src_len = batch["src_len"].to(device)
        y_out = batch["y_out"]  # cpu reference ids

        # candidates
        if decode == "greedy":
            pred_ids = decode_greedy(
                encoder, decoder, src, src_len,
                max_len=y_out.size(1) + 10, sos_id=sos, eos_id=eos
            )
            cand = ids_to_tokens(pred_ids, itos_en)
        else:
            cand = []
            for i in range(src.size(0)):
                seq = beam_search(
                    encoder, decoder, src[i:i+1], src_len[i:i+1],
                    max_len=y_out.size(1) + 10, sos_id=sos, eos_id=eos, beam=beam_size
                )
                cand.append([itos_en[t] for t in seq])

        # references
        refs = []
        for row in y_out.tolist():
            toks = []
            for t in row:
                if t == pad:
                    break
                if t == sos:
                    continue
                if t == eos:
                    break
                toks.append(itos_en[t])
            refs.append(toks)

        all_cand += cand
        all_ref += refs

    return corpus_bleu(all_cand, all_ref, max_n=4)


def load_checkpoint(path, encoder, decoder, opt=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
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

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--rnn_type", choices=["gru", "lstm"], default="lstm")
    ap.add_argument("--attn", choices=["dot", "general", "additive"], default="dot")
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--free_running", action="store_true")

    # NEW: decode strategy split (fast training + final report)
    ap.add_argument("--beam_size", type=int, default=4)
    ap.add_argument("--train_decode", choices=["greedy", "beam"], default="greedy",
                    help="decoding used during training/valid monitoring (recommend greedy)")
    ap.add_argument("--final_decode", choices=["none", "greedy", "beam"], default="beam",
                    help="extra decoding after training for reporting")
    ap.add_argument("--final_beam_size", type=int, default=4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_root", type=str, default=None)

    ap.add_argument("--resume", type=str, default=None, help="path to best.pt/last.pt")
    ap.add_argument("--eval_test_each_epoch", action="store_true", help="also evaluate test each epoch (slow)")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # paths
    save_root = args.save_root or os.path.join(args.data_dir, "checkpoints")
    run_name = make_run_name_rnn(args)
    ckpt_dir = ensure_dir(os.path.join(save_root, "rnn", run_name))

    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    log_path = os.path.join(ckpt_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,valid_bleu4,test_bleu4,seconds\n")

    # vocab
    _, zh_itos = load_vocab(args.vocab_zh)
    _, en_itos = load_vocab(args.vocab_en)
    Vsrc, Vtgt = len(zh_itos), len(en_itos)

    # data
    train_loader = make_loader(f"{args.data_dir}/train_tok.jsonl", args.batch_size, shuffle=True)
    valid_loader = make_loader(f"{args.data_dir}/valid_tok.jsonl", args.batch_size, shuffle=False)
    test_loader  = make_loader(f"{args.data_dir}/test_tok.jsonl",  args.batch_size, shuffle=False)

    # model
    encoder = Encoder(Vsrc, args.emb, args.hidden, num_layers=2, rnn_type=args.rnn_type).to(device)
    decoder = Decoder(Vtgt, args.emb, args.hidden, num_layers=2, rnn_type=args.rnn_type, attn_type=args.attn).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=SPECIALS["<PAD>"])
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    best_bleu = -1.0
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")

    start_epoch = 0
    if args.resume:
        start_epoch, best_bleu = load_checkpoint(args.resume, encoder, decoder, opt=opt, device=device)
        print(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_bleu={best_bleu:.4f})")

    # train
    for ep in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        tf = not args.free_running

        train_loss = train_one_epoch(
            encoder, decoder, train_loader, opt, loss_fn, device,
            teacher_forcing=tf
        )

        valid_bleu = evaluate_bleu(
            encoder, decoder, valid_loader, en_itos, device,
            decode=args.train_decode, beam_size=args.beam_size
        )

        test_bleu = float("nan")
        if args.eval_test_each_epoch:
            test_bleu = evaluate_bleu(
                encoder, decoder, test_loader, en_itos, device,
                decode=args.train_decode, beam_size=args.beam_size
            )

        dt = time.time() - t0
        print(f"[Epoch {ep}] train_loss={train_loss:.4f}  valid_BLEU4={valid_bleu*100:.2f}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ep},{train_loss:.6f},{valid_bleu*100:.4f},{test_bleu*100 if test_bleu==test_bleu else 'nan'},{dt:.2f}\n")

        # save last
        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
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
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "opt": opt.state_dict(),
                "config": vars(args),
                "epoch": ep,
                "valid_bleu4": valid_bleu,
                "best_bleu": best_bleu
            }, best_path)

    # -------------------------
    # final reporting (fast train + final decode)
    # -------------------------
    # Load best checkpoint
    if os.path.exists(best_path):
        load_checkpoint(best_path, encoder, decoder, opt=None, device=device)

    # Always compute train_decode metrics on best
    best_valid_trainDec = evaluate_bleu(
        encoder, decoder, valid_loader, en_itos, device,
        decode=args.train_decode, beam_size=args.beam_size
    )
    best_test_trainDec = evaluate_bleu(
        encoder, decoder, test_loader, en_itos, device,
        decode=args.train_decode, beam_size=args.beam_size
    )

    extra = {}
    if args.final_decode != "none":
        extra_valid = evaluate_bleu(
            encoder, decoder, valid_loader, en_itos, device,
            decode=args.final_decode, beam_size=args.final_beam_size
        )
        extra_test = evaluate_bleu(
            encoder, decoder, test_loader, en_itos, device,
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
