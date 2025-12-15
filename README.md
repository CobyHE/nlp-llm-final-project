# NMT Final Project – Inference Guide

This repository contains three translation systems:

1. **RNN-based NMT (Encoder–Decoder + Attention)**  
2. **Transformer-based NMT (trained from scratch)**  
3. **mT5-small fine-tuned model (pretrained LM adaptation)**

We provide a one-click inference script: **`inference.py`**.

---

## 1) Environment

Recommended: Python 3.11 + PyTorch + Transformers (for mT5).

Install dependencies (example):

```bash
pip install torch transformers sentencepiece
```

> If you only run RNN/Transformer inference, transformers is not strictly required.
## Directory Layout (Example)
Checkpoints are stored under:
```text
nmt_data_jieba_100k/
  checkpoints/
    rnn/
      <rnn_run_name>/
        best.pt
        last.pt
        config.json
        summary.json
        train_log.csv
    transformer/
      <tfm_run_name>/
        best.pt
        last.pt
        config.json
        summary.json
        train_log.csv
```
Vocabulary files:
```text
nmt_data_jieba_100k/vocab_zh.json
nmt_data_jieba_100k/vocab_en.json
```

Local pretrained model directory (example):
```text
/root/LLMs/mt5-small
```
## 3) Quick Start: Translate One Sentence
### A) RNN checkpoint (greedy / beam)

Greedy decoding:
```bash
python inference.py \
  --model_type rnn \
  --checkpoint /root//final_project/nmt_data_jieba_100k/checkpoints/rnn/<RUN_NAME>/best.pt \
  --vocab_zh /root/final_project/nmt_data_jieba_100k/vocab_zh.json \
  --vocab_en /root/final_project/nmt_data_jieba_100k/vocab_en.json \
  --decode greedy \
  --text "1990 年 ？"
```

Beam search (slower but sometimes better):
```bash
python inference.py \
  --model_type rnn \
  --checkpoint /root/final_project/nmt_data_jieba_100k/checkpoints/rnn/<RUN_NAME>/best.pt \
  --vocab_zh /root/final_project/nmt_data_jieba_100k/vocab_zh.json \
  --vocab_en /root/final_project/nmt_data_jieba_100k/vocab_en.json \
  --decode beam --beam_size 4 \
  --text "1990 年 ？"
```
### B) Transformer checkpoint (greedy / beam)

Greedy decoding:
```bash
python inference.py \
  --model_type transformer \
  --checkpoint /root/final_project/nmt_data_jieba_100k/checkpoints/transformer/<RUN_NAME>/best.pt \
  --vocab_zh /root/final_project/nmt_data_jieba_100k/vocab_zh.json \
  --vocab_en /root/final_project/nmt_data_jieba_100k/vocab_en.json \
  --decode greedy \
  --text "1990 年 ？"
```

Beam search:
```bash
python inference.py \
  --model_type transformer \
  --checkpoint /root/final_project/nmt_data_jieba_100k/checkpoints/transformer/<RUN_NAME>/best.pt \
  --vocab_zh /root/final_project/nmt_data_jieba_100k/vocab_zh.json \
  --vocab_en /root/final_project/nmt_data_jieba_100k/vocab_en.json \
  --decode beam --beam_size 4 \
  --text "1990 年 ？"
```
### C) mT5 (pretrained language model)

For mT5 inference, only --mt5_dir and input text are required:
```bash
python inference.py \
  --model_type mt5 \
  --mt5_dir /root/LLMs/mt5-small \
  --text "1990 年 ？"
```

Optional beam search:
```bash
python inference.py \
  --model_type mt5 \
  --mt5_dir /root/LLMs/mt5-small \
  --decode beam --beam_size 4 \
  --text "1990 年 ？"
```

> **Note**: mT5 may output special placeholder tokens such as <extra_id_0> on very short or ambiguous inputs. This is expected behavior for T5-family models under limited fine-tuning.

## 4) Batch Translation (File → File)

Prepare a plain text file with one Chinese sentence per line:

```text
1990 年 ？
广州 是 一 座 城市
...
```

Run batch inference:
```bash
python inference.py \
  --model_type rnn \
  --checkpoint /root/Course/final_project/nmt_data_jieba_100k/checkpoints/rnn/<RUN_NAME>/best.pt \
  --vocab_zh /root/Course/final_project/nmt_data_jieba_100k/vocab_zh.json \
  --vocab_en /root/Course/final_project/nmt_data_jieba_100k/vocab_en.json \
  --decode greedy \
  --input_file input.txt \
  --output_file output.txt
```

The same interface applies to transformer and mt5 models (for mT5, use --mt5_dir instead of --checkpoint and vocab files).

## 5) Useful Arguments

+ --decode {greedy,beam}: decoding strategy (default: greedy)

+ --beam_size N: beam size (default: 4)

+ --max_gen N: maximum generated length (default: 80)

+ --device cpu|cuda: manually set device (default: auto)

+ --seed N: random seed for reproducibility

## 6) Available Checkpoints
**RNN models** (`checkpoints/rnn/`)

+ GRU / LSTM encoders

+ Attention types: dot, general, additive

+ Training policies: teacher forcing vs free running

Each run directory contains:

+ `best.pt`: best validation checkpoint

+ `last.pt`: last epoch checkpoint

+ `config.json`, `summary.json`, `train_log.csv`

**Transformer models** (`checkpoints/transformer/`)

+ Position embeddings: learned, sinusoidal

+ Normalization: LayerNorm, RMSNorm

+ Multiple model sizes, batch sizes, and learning rates

Use best.pt in any run directory for inference.
