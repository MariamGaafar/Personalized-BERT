"""
04_compute_bert_embeddings.py
------------------------------
Compute BERT-Base-Cased mean-pooled embeddings for every tweet in
pan17_documents.json and write them to a JSONL file.

Model: bert-base-cased  (12 layers, 768-d hidden, no lower-casing)
Pooling: mean of the last hidden state across all non-padding tokens
         (CLS and SEP tokens are included in the mean, matching the paper).

Input
-----
    data/pan17_documents.json   {author_id: [tweet_str x 100]}

Output
------
    data/Pan17_bert_embeddings.jsonl
    Each line: {author_id: [[768 floats] x 100]}
"""

import argparse
import json
import os

import torch
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME  = "bert-base-cased"
BATCH_SIZE  = 32          # tweets per forward pass
MAX_LENGTH  = 128         # max token length per tweet
IN_PATH     = "data/pan17_documents.json"
OUT_PATH    = "data/Pan17_bert_embeddings.jsonl"


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool the last hidden states, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    count  = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


@torch.no_grad()
def embed_batch(texts: list, tokenizer, model, device: torch.device) -> list:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    out = model(**enc)
    pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
    return pooled.cpu().tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_embeddings(
    in_path:  str = IN_PATH,
    out_path: str = OUT_PATH,
    batch_size: int = BATCH_SIZE,
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        documents: dict = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = len(documents)
    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, (author_id, tweets) in enumerate(documents.items()):
            author_embeddings: list = []

            # Process in batches
            for start in range(0, len(tweets), batch_size):
                batch = tweets[start : start + batch_size]
                vecs  = embed_batch(batch, tokenizer, model, device)
                author_embeddings.extend(vecs)

            record = {author_id: author_embeddings}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                print(f"  {idx+1}/{total} authors processed")

    print(f"\nEmbeddings saved to '{out_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BERT embeddings for PAN-17")
    parser.add_argument("--input",      type=str, default=IN_PATH,   help="Path to pan17_documents.json")
    parser.add_argument("--output",     type=str, default=OUT_PATH,  help="Output JSONL path")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Tweets per forward pass")
    args = parser.parse_args()
    compute_embeddings(args.input, args.output, args.batch_size)
