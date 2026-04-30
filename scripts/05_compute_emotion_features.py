"""
05_compute_emotion_features.py
-------------------------------
Run the DistilRoBERTa emotion classifier on every tweet in
pan17_documents.json and save the 6-d probability vectors.

Model : michellejieli/emotion_text_classifier
        (fine-tuned on 7 emotion datasets; labels: anger, disgust, fear,
         joy, neutral, sadness, surprise)

Input
-----
    data/pan17_documents.json   {author_id: [tweet_str x 100]}

Output
------
    data/pan17_emotion_predictions.jsonl
    Each line: {tweet_key: [p_anger, p_disgust, p_fear, p_joy, p_neutral, p_sadness, p_surprise]}

    tweet_key format: "{author_id}_{tweet_index}"   (0-based index)
"""

import argparse
import json
import os

import torch
from transformers import pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME  = "michellejieli/emotion_text_classifier"
BATCH_SIZE  = 64
IN_PATH     = "data/pan17_documents.json"
OUT_PATH    = "data/pan17_emotion_predictions.jsonl"

# Canonical label order (sorted alphabetically to guarantee consistent indexing)
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_emotion_features(
    in_path:    str = IN_PATH,
    out_path:   str = OUT_PATH,
    batch_size: int = BATCH_SIZE,
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        documents: dict = json.load(f)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_NAME} …  (device={'GPU' if device==0 else 'CPU'})")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,          # return all label scores
        device=device,
        truncation=True,
        max_length=128,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total_authors = len(documents)

    with open(out_path, "w", encoding="utf-8") as fout:
        for a_idx, (author_id, tweets) in enumerate(documents.items()):
            batch_record: dict = {}

            # Process in batches
            for start in range(0, len(tweets), batch_size):
                batch      = tweets[start : start + batch_size]
                # Replace empty strings to avoid tokenizer warnings
                safe_batch = [t if t.strip() else "." for t in batch]
                outputs    = classifier(safe_batch)

                for rel_idx, label_scores in enumerate(outputs):
                    tweet_idx = start + rel_idx
                    tweet_key = f"{author_id}_{tweet_idx}"

                    # Map label→score, then build ordered vector
                    score_map = {ls["label"].lower(): ls["score"] for ls in label_scores}
                    vector    = [score_map.get(lbl, 0.0) for lbl in EMOTION_LABELS]
                    batch_record[tweet_key] = vector

            fout.write(json.dumps(batch_record, ensure_ascii=False) + "\n")

            if (a_idx + 1) % 100 == 0 or (a_idx + 1) == total_authors:
                print(f"  {a_idx+1}/{total_authors} authors processed")

    print(f"\nEmotion predictions saved to '{out_path}'")
    print(f"Label order: {EMOTION_LABELS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute emotion features for PAN-17")
    parser.add_argument("--input",      type=str, default=IN_PATH,   help="Path to pan17_documents.json")
    parser.add_argument("--output",     type=str, default=OUT_PATH,  help="Output JSONL path")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Tweets per batch")
    args = parser.parse_args()
    compute_emotion_features(args.input, args.output, args.batch_size)
